"""Hand-checked tests for the nested-JSON parser and a small fixture chunk."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from analysis import loader


# ---- _safe_parse: datetime preprocessing + ast.literal_eval -----------

def test_safe_parse_simple_dict():
    s = "{'a': 1, 'b': 'two', 'c': True, 'd': None}"
    assert loader._safe_parse(s) == {"a": 1, "b": "two", "c": True, "d": None}


def test_safe_parse_list_of_dicts():
    s = "[{'x': 1}, {'x': 2}]"
    assert loader._safe_parse(s) == [{"x": 1}, {"x": 2}]


def test_safe_parse_none_and_empty():
    assert loader._safe_parse(None) is None
    assert loader._safe_parse("") is None
    assert loader._safe_parse(float("nan")) is None  # type: ignore[arg-type]


def test_safe_parse_user_with_datetime_utc():
    s = (
        "{'id': 1, 'created': datetime.datetime(2017, 12, 18, 21, 28, 43, "
        "tzinfo=datetime.timezone.utc), 'verified': False}"
    )
    parsed = loader._safe_parse(s)
    assert parsed is not None
    assert parsed["id"] == 1
    assert parsed["created"] == "2017-12-18T21:28:43+00:00"
    assert parsed["verified"] is False


def test_safe_parse_user_with_microseconds():
    s = "{'created': datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)}"
    parsed = loader._safe_parse(s)
    assert parsed is not None
    assert parsed["created"] == "2020-01-02T03:04:05.123456+00:00"


def test_safe_parse_malformed_returns_none():
    # Unterminated string — should NOT raise
    assert loader._safe_parse("{'a': 'unterm") is None


# ---- field extractors -------------------------------------------------

def test_extract_user_fields_full_dict():
    user = {
        "id_str": "1",
        "username": "alice",
        "followersCount": 100,
        "friendsCount": 50,
        "statusesCount": 10,
        "favouritesCount": 5,
        "listedCount": 0,
        "mediaCount": 1,
        "verified": False,
        "blue": True,
        "created": "2018-01-01T00:00:00+00:00",
    }
    out = loader._extract_user_fields(user)
    assert out["user_id_str"] == "1"
    assert out["user_followersCount"] == 100
    assert out["user_blue"] is True
    assert out["user_created"] == "2018-01-01T00:00:00+00:00"


def test_extract_user_fields_none_returns_nulls():
    out = loader._extract_user_fields(None)
    for k in loader.USER_FIELDS:
        assert out[f"user_{k}"] is None


def test_parse_view_count_normal():
    assert loader._parse_view_count({"count": "1234", "state": "EnabledWithCount"}) == 1234


def test_parse_view_count_edge_cases():
    assert loader._parse_view_count(None) is None
    assert loader._parse_view_count({"count": None}) is None
    assert loader._parse_view_count({"count": "None"}) is None
    assert loader._parse_view_count({"count": "abc"}) is None
    assert loader._parse_view_count({"state": "EnabledWithCount"}) is None  # missing count


def test_extract_link_urls_picks_expanded():
    links = [
        {"display_url": "ex.com/a", "expanded_url": "https://example.com/a", "url": "https://t.co/x"},
        {"display_url": "ex.com/b", "expanded_url": "https://example.com/b", "url": "https://t.co/y"},
    ]
    assert loader._extract_link_urls(links) == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_extract_link_urls_empty_and_malformed():
    assert loader._extract_link_urls(None) == []
    assert loader._extract_link_urls([]) == []
    assert loader._extract_link_urls([{"url": "https://t.co/x"}]) == []  # no expanded
    assert loader._extract_link_urls(["not a dict"]) == []


# ---- end-to-end on the real chunk -------------------------------------

CHUNK_PATH = Path("data/raw/usc-x-24/part_1/may_july_chunk_1.csv.gz")


@pytest.mark.skipif(not CHUNK_PATH.exists(), reason="USC chunk not present")
def test_load_chunk_smoke():
    df = loader.load_chunk(CHUNK_PATH).head(500)  # any subset is fine
    # Required columns
    for col in (
        "id", "id_str", "epoch", "created_at", "rawContent",
        "replyCount", "retweetCount", "likeCount", "quoteCount",
        "view_count", "link_urls", "is_reply", "is_quote",
        "is_retweet", "is_original", "user_followersCount",
        "user_username", "user_verified", "user_created_at",
        "conversationId", "in_reply_to_status_id_str",
    ):
        assert col in df.columns, f"missing column {col}"

    # is_* flags should be mutually informative
    assert df["is_reply"].dtype == bool
    assert df["is_quote"].dtype == bool
    assert df["is_retweet"].dtype == bool

    # link_urls is a list-typed column; lengths should be sensible
    lengths = df["link_urls"].map(len)
    assert lengths.min() >= 0
    # At least some tweets in the first 500 should have URLs
    assert lengths.max() >= 1, "expected at least one tweet with a URL"

    # user_created_at should parse to a valid datetime for most rows
    assert df["user_created_at"].notna().mean() > 0.9

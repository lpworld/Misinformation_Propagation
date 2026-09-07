"""Tests for credibility labeling: domain extraction and bucket assignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis import labeling


# ---- domain extraction edge cases ------------------------------------

def test_url_to_registered_domain_simple():
    assert labeling.url_to_registered_domain("https://example.com/path") == "example.com"


def test_url_to_registered_domain_subdomain():
    assert labeling.url_to_registered_domain("https://news.example.com/x") == "example.com"


def test_url_to_registered_domain_two_part_tld():
    assert labeling.url_to_registered_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"


def test_url_to_registered_domain_uppercase():
    assert labeling.url_to_registered_domain("HTTPS://EXAMPLE.COM/A") == "example.com"


def test_url_to_registered_domain_no_scheme():
    assert labeling.url_to_registered_domain("example.com/path") == "example.com"


def test_url_to_registered_domain_invalid():
    assert labeling.url_to_registered_domain("") == ""
    assert labeling.url_to_registered_domain("not a url") == ""
    assert labeling.url_to_registered_domain(None) == ""  # type: ignore[arg-type]


# ---- label_urls bucket logic ----------------------------------------

IFFY = {"badnews.com", "fakeoutlet.org"}
MAINSTREAM = {"nytimes.com", "bbc.co.uk"}


def test_label_urls_low_only():
    assert labeling.label_urls(["https://badnews.com/x"], IFFY, MAINSTREAM) == labeling.LOW_CRED


def test_label_urls_high_only():
    assert labeling.label_urls(["https://www.nytimes.com/x"], IFFY, MAINSTREAM) == labeling.HIGH_CRED


def test_label_urls_mixed():
    urls = ["https://badnews.com/x", "https://www.bbc.co.uk/y"]
    assert labeling.label_urls(urls, IFFY, MAINSTREAM) == labeling.MIXED


def test_label_urls_no_match():
    assert labeling.label_urls(["https://random-site.io/x"], IFFY, MAINSTREAM) is None


def test_label_urls_empty():
    assert labeling.label_urls([], IFFY, MAINSTREAM) is None
    assert labeling.label_urls(None, IFFY, MAINSTREAM) is None


def test_label_urls_subdomain_matches_registered_domain():
    # A subdomain like 'm.bbc.co.uk' should still match the 'bbc.co.uk' entry
    assert labeling.label_urls(
        ["https://m.bbc.co.uk/x"], IFFY, MAINSTREAM
    ) == labeling.HIGH_CRED


# ---- list loaders ---------------------------------------------------

def test_load_mainstream_domains_skips_comments(tmp_path: Path):
    p = tmp_path / "ms.txt"
    p.write_text(
        "# header\n"
        "  # leading whitespace then comment\n"
        "\n"
        "Example.COM\n"
        "  bbc.co.uk  # inline comment is stripped\n"
        "another.org\n",
        encoding="utf-8",
    )
    out = labeling.load_mainstream_domains(p)
    assert out == {"example.com", "bbc.co.uk", "another.org"}


def test_load_iffy_domains_real_file():
    # uses the actual downloaded list
    p = Path("data/labels/iffy_plus.csv")
    if not p.exists():
        pytest.skip("Iffy+ CSV not present")
    out = labeling.load_iffy_domains(p)
    assert len(out) > 100
    # spot-check a known entry seen during probe
    assert "100percentfedup.com" in out


# ---- add_labels on a tiny DataFrame ---------------------------------

def test_add_labels_dataframe(tmp_path: Path):
    iffy = tmp_path / "iffy.csv"
    iffy.write_text("Domain\nbadnews.com\n", encoding="utf-8")
    ms = tmp_path / "ms.txt"
    ms.write_text("nytimes.com\n", encoding="utf-8")
    df = pd.DataFrame(
        {
            "id_str": ["1", "2", "3", "4"],
            "link_urls": [
                ["https://badnews.com/x"],
                ["https://www.nytimes.com/x"],
                ["https://random.io/x"],
                [],
            ],
        }
    )
    out = labeling.add_labels(df, iffy_path=iffy, mainstream_path=ms)
    assert list(out["credibility_label"]) == [
        labeling.LOW_CRED,
        labeling.HIGH_CRED,
        None,
        None,
    ]

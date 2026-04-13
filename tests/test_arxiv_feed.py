"""Tests for src/arxiv_feed.py"""

import textwrap

import pytest

from src.arxiv_feed import _parse_api_response, extract_arxiv_id, _parse_entry
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# extract_arxiv_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("http://arxiv.org/abs/2401.12345v2", "2401.12345v2"),
        ("oai:arXiv.org:2401.12345", "2401.12345"),
        ("2401.12345", "2401.12345"),
        ("2401.12345v3", "2401.12345v3"),
        ("", None),
        ("not-an-id", None),
    ],
)
def test_extract_arxiv_id(raw, expected):
    assert extract_arxiv_id(raw) == expected


# ---------------------------------------------------------------------------
# _parse_api_response – full XML roundtrip
# ---------------------------------------------------------------------------

SAMPLE_ATOM = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.99999v1</id>
        <title>A Test Paper on Quantum Computing</title>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Jones</name></author>
        <summary>  We present a test paper.  </summary>
        <published>2024-01-15T00:00:00Z</published>
        <category term="quant-ph"/>
        <category term="cs.ET"/>
      </entry>
    </feed>
    """
)


def test_parse_api_response_basic():
    papers = _parse_api_response(SAMPLE_ATOM)
    assert len(papers) == 1
    p = papers[0]
    assert p.paper_id == "2401.99999v1"
    assert "Quantum Computing" in p.title
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert "test paper" in p.abstract
    assert p.url == "https://arxiv.org/abs/2401.99999v1"
    assert p.date_published == "2024-01-15T00:00:00Z"
    assert "quant-ph" in p.categories
    assert "cs.ET" in p.categories


def test_parse_api_response_normalises_whitespace():
    xml = textwrap.dedent(
        """\
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2401.00001</id>
            <title>Title with
        newline</title>
            <summary>Abstract with
        newline</summary>
          </entry>
        </feed>
        """
    )
    papers = _parse_api_response(xml)
    assert len(papers) == 1
    assert "\n" not in papers[0].title
    assert "\n" not in papers[0].abstract


def test_parse_api_response_empty_feed():
    xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    papers = _parse_api_response(xml)
    assert papers == []


def test_parse_api_response_invalid_xml():
    papers = _parse_api_response("<<not xml>>")
    assert papers == []


def test_parse_api_response_multiple_entries():
    xml = textwrap.dedent(
        """\
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2401.00001</id>
            <title>Paper One</title>
            <summary>Abstract one.</summary>
          </entry>
          <entry>
            <id>http://arxiv.org/abs/2401.00002</id>
            <title>Paper Two</title>
            <summary>Abstract two.</summary>
          </entry>
        </feed>
        """
    )
    papers = _parse_api_response(xml)
    assert len(papers) == 2
    titles = {p.title for p in papers}
    assert "Paper One" in titles
    assert "Paper Two" in titles

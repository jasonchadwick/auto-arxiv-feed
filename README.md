# auto-arxiv-feed

Automated tool that monitors arXiv for new papers and emails a daily digest of
those that are relevant to your research, determined by comparing embedding
vectors against your personal Zotero library.

---

## How it works

1. **Hourly (Zotero sync)** – `scripts/update_zotero.py` queries the Zotero
   API, stores new papers in a local SQLite database, and computes embeddings
   for any papers not yet embedded.

2. **Daily (arXiv digest)** – `scripts/daily_digest.py` reads today's papers
   from the arXiv RSS feeds you've configured, embeds each one, computes cosine
   similarity against every Zotero paper embedding, and emails (or prints) a
   digest of papers above your chosen similarity threshold.

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml – fill in Zotero credentials, arXiv categories,
# embedding provider + API key, and email settings.
```

### 3. Populate the database

```bash
python scripts/update_zotero.py --config config.yaml
```

Run this on a schedule (e.g. every hour via cron):

```
0 * * * * cd /path/to/auto-arxiv-feed && .venv/bin/python scripts/update_zotero.py
```

### 4. Send the daily digest

```bash
python scripts/daily_digest.py --config config.yaml
```

Run this once per day, shortly after arXiv's daily announcement:

```
30 14 * * * cd /path/to/auto-arxiv-feed && .venv/bin/python scripts/daily_digest.py
```

### Dry-run mode

Both scripts accept `--dry-run` to preview what would happen without calling
any external API or sending email.  The digest script also accepts `--threshold`
so you can experiment with different cut-offs:

```bash
python scripts/daily_digest.py --dry-run --threshold 0.75
```

---

## Embedding providers

| Provider  | Config value | Environment variable | Notes |
|-----------|-------------|----------------------|-------|
| OpenAI    | `openai`    | `OPENAI_API_KEY`     | `text-embedding-3-small` / `text-embedding-3-large` |
| Gemini    | `gemini`    | `GEMINI_API_KEY`     | `models/text-embedding-004` |
| Local     | `local`     | –                    | Subclass `LocalEmbedder` and implement `_load_model()` / `embed()` |
| Anthropic | `anthropic` | –                    | **Not available** – Anthropic has no public embedding API yet |

---

## Project layout

```
auto-arxiv-feed/
├── config.yaml.example        # template – copy to config.yaml
├── requirements.txt
├── src/
│   ├── arxiv_feed.py          # arXiv RSS + Atom API
│   ├── database.py            # SQLite paper & embedding store
│   ├── email_digest.py        # HTML/plain-text email builder + sender
│   ├── embeddings.py          # provider implementations
│   ├── relevance.py           # cosine similarity filtering
│   └── zotero_client.py       # Zotero API wrapper
├── scripts/
│   ├── update_zotero.py       # hourly: sync Zotero & embed new papers
│   └── daily_digest.py        # daily: arXiv → relevance check → email
└── tests/
    ├── test_arxiv_feed.py
    ├── test_database.py
    ├── test_embeddings.py
    └── test_relevance.py
```

---

## Running the tests

```bash
pytest tests/
```

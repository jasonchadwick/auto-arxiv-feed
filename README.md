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
   from the arXiv RSS feeds you've configured, embeds each one, computes a
   continuous LOF density score against your Zotero embedding distribution, and
   emails (or prints) a digest of papers above your chosen threshold.

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

Relevance can also be constrained with a sanity-check include list:

```yaml
relevance:
   threshold: 0.50
   lof_neighbors: 20
   top_k: 5
   always_include_terms: ["quantum error correction", "superconducting qubit"]
```

Any new paper whose title or abstract includes one of these terms is always
included in the digest, even if its similarity score is below threshold.

### 3. Populate the database

```bash
python scripts/update_zotero.py --config config.yaml
```

Run this on a schedule (e.g. every hour via cron):

```
0 * * * * cd /path/to/auto-arxiv-feed && .venv/bin/python scripts/update_zotero.py
```

For transient network outages, you can enable retry-until-success in
config.yaml:

```yaml
resilience:
   retry_until_success: true
   retry_delay_seconds: 30
   retry_backoff: 2.0
   max_retry_delay_seconds: 600
```

### 4. Send the daily digest

```bash
python scripts/daily_digest.py --config config.yaml
```

Run this once per day, shortly after arXiv's daily announcement:

```
30 14 * * * cd /path/to/auto-arxiv-feed && .venv/bin/python scripts/daily_digest.py
```

When a script fails, it attempts to send an error email with retries. If SMTP
is unreachable, the unsent notification is persisted to
log/unsent_error_notifications.log so failures are not silent.

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
| Local     | `local`     | –                    | Built-in `sentence-transformers` support (no API key required) |
| Anthropic | `anthropic` | –                    | **Not available** – Anthropic has no public embedding API yet |

---

## Choosing a local embedding model

When `embedding.provider` is set to `local`, the project uses
`sentence-transformers` and loads your chosen HuggingFace model name from
`embedding.model`.

### Install options

CPU-only:

```bash
pip install sentence-transformers
```

NVIDIA GPU on Linux or WSL2:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers
```

On WSL2, CUDA works through the Windows NVIDIA driver passthrough; you do not
need a separate Linux CUDA toolkit install for this project.

### Model recommendations

| Model | Dimensions | Approx size | Best for |
|-------|------------|-------------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | ~133 MB | Default choice: best speed/quality balance |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3 GB | Highest quality, slower and heavier |
| `allenai-specter` | 768 | ~400 MB | Scientific-paper similarity (often strong for arXiv workflows) |
| `all-MiniLM-L6-v2` | 384 | ~22 MB | Fastest and lightest, lower retrieval quality |

### How to choose

1. Start with `BAAI/bge-small-en-v1.5`.
2. If relevance quality is too low, try `allenai-specter` (domain-specific) or
   `BAAI/bge-large-en-v1.5` (general high quality).
3. If runtime/memory is your main constraint, use `all-MiniLM-L6-v2`.
4. Re-tune your similarity threshold after changing models (for example with
   `scripts/daily_digest.py --dry-run --threshold 0.75`) because score
   distributions differ by model.

### Config example

```yaml
embedding:
  provider: local
  model: BAAI/bge-small-en-v1.5
```

First run downloads the model to your HuggingFace cache; later runs reuse it.

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
│   ├── relevance.py           # centroid-similarity relevance filtering
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

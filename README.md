# ResearchPaperLens

A small, modular pipeline that reads a research-paper PDF, extracts structured
metadata (title, authors, abstract, section headings, and more), optionally
generates a structured LLM summary using an Ollama cloud model, and saves the
result as clean JSON.

## Current Phase

ResearchPaperLens extracts structured metadata from research-paper PDFs, saves
the result as JSON, optionally generates an LLM summary, and supports semantic
retrieval plus RAG Q&A over indexed paper chunks.

Web deployment is not implemented yet.

## Features

- Extracts text from research-paper PDFs
- Counts pages and words
- Guesses title and authors
- Extracts abstract and keywords when present
- Detects section headings
- Word-based chunking for long papers (`src/chunking.py`), with configurable
  overlap between chunks
- Optional LLM summarization: fills a structured `summary` (tldr, problem,
  approach, key results/metrics, contributions, limitations, key insights) via
  an Ollama cloud model
  - `--summarize` — single-pass summary (first ~12k characters of the paper)
  - `--summarize-full` — section-aware hybrid summarization: split by section,
    word-chunk long sections, merge section summaries into a final summary
    (recommended for long papers)
  - Model selectable with `--model`
- Semantic retrieval: hybrid-chunk embeddings via Ollama (`nomic-embed-text`),
  per-paper chunk JSON stores, in-memory cosine search, CLI `--index` /
  `--search`
- RAG Q&A: retrieve top-k chunks, answer questions with Ollama cloud (`--ask`),
  with section citations and source chunk ids
- Saves structured results as JSON
- Includes unit and integration tests

## How it works

The pipeline is split into focused, independently testable modules:

```
.
├── src/
│   ├── paper.py        # ResearchPaper data model (dataclass)
│   ├── pdf_reader.py   # PDF I/O: extract full text + page count
│   ├── analyzer.py     # Pure analysis: text -> ResearchPaper fields
│   ├── chunking.py     # Word-based text chunking with overlap
│   ├── ollama_client.py # Shared Ollama cloud client (API key)
│   ├── embeddings.py   # Ollama cloud text embeddings
│   ├── chunk_store.py  # Indexed chunks + JSON persistence
│   ├── vector_index.py # In-memory cosine similarity search
│   ├── retrieval.py    # index_paper() and search() orchestration
│   ├── rag.py          # RAG Q&A over retrieved chunks
│   ├── summarizer.py   # LLM summarizer (Ollama cloud): summary/key_insights
│   └── storage.py      # Serialize ResearchPaper <-> JSON
├── data/               # Input PDFs
├── outputs/            # Generated JSON output (git-ignored)
├── tests/              # Unit tests + tests/integration/ end-to-end test
├── main.py             # CLI entry point that wires the pipeline together
├── requirements.txt    # Python dependencies
└── README.md
```

Data flows in one direction:

```
PDF --(pdf_reader)--> text + page count --(analyzer)--> ResearchPaper --(storage)--> JSON
                                                              |
                                         optional --(summarizer)--> summary
                                                              |
                              --summarize-full--> sections --> per-section summaries --> merge
```

`analyzer.py` is deliberately decoupled from PDF reading: it only accepts
already-extracted text, which keeps the analysis logic easy to test and reusable
for any text source.

For `--summarize-full`, `chunking.py` splits the paper into top-level sections
(Abstract, Preamble, numbered sections). Sections longer than 1,200 words are
subdivided with fixed-word chunks (with overlap). The summarizer summarizes each
section (merging sub-chunks when needed), then merges section summaries into the
final structured summary using paper metadata as anchors.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Summarization (optional)

LLM summarization uses [Ollama cloud models](https://docs.ollama.com/cloud) via
the cloud API. To use `--summarize` or `--summarize-full`, create an API key at
[ollama.com/settings/keys](https://ollama.com/settings/keys) and provide it as
the `OLLAMA_API_KEY` environment variable.

The recommended way is a local `.env` file (auto-loaded via `python-dotenv` and
git-ignored, so your key is never committed). Copy the template and fill it in:

```bash
cp .env.example .env
# then edit .env and set OLLAMA_API_KEY=your_api_key
```

Alternatively, export it in your shell:

```bash
export OLLAMA_API_KEY=your_api_key
```

The key is only required when running with `--summarize` or `--summarize-full`;
the rest of the pipeline works without it.

## Usage

Run the pipeline on a PDF:

```bash
# Use the bundled sample paper -> outputs/AttentionIsAllYouNeed.json
python main.py

# Analyze your own PDF (output defaults to outputs/<pdf-stem>.json)
python main.py path/to/paper.pdf

# Choose an explicit output file
python main.py path/to/paper.pdf -o results/paper.json

# Summarize with an Ollama cloud model (requires OLLAMA_API_KEY)
python main.py path/to/paper.pdf --summarize

# Full-paper summarization via chunking + merge (recommended for long papers)
python main.py path/to/paper.pdf --summarize-full

# Switch the cloud model used for summarization
python main.py path/to/paper.pdf --summarize-full --model gpt-oss:20b

# Build a semantic chunk index (embed + save outputs/<pdf-stem>.chunks.json)
python main.py path/to/paper.pdf --index

# Search an existing chunk index
python main.py --search "What BLEU scores were reported?" \
  --from outputs/AttentionIsAllYouNeed.chunks.json

# Show more search hits
python main.py --search "multi-head attention" \
  --from outputs/AttentionIsAllYouNeed.chunks.json --top-k 10

# Ask a grounded question (retrieve + Ollama cloud answer)
python main.py --ask "How does multi-head attention work?" \
  --from outputs/AttentionIsAllYouNeed.chunks.json
```

See all options with `python main.py --help`.

> **Note:** `--summarize` and `--summarize-full` are mutually exclusive. Both
> require `OLLAMA_API_KEY`. The model defaults to `gpt-oss:120b` and can be
> changed with `--model`; other options include `gpt-oss:20b`, `qwen3-coder:480b`,
> and `deepseek-v3.1:671b` (see
> [Ollama cloud models](https://ollama.com/search?c=cloud)).
>
> - **`--summarize`** — one API call; sends only the first ~12k characters of
>   the paper (fast, but may miss later sections on long papers).
> - **`--summarize-full`** — hierarchical hybrid summarization (default
>   strategy): split by section, word-chunk sections over 1,200 words, summarize
>   each section, merge into a final summary. Multiple API calls (more than
>   `--summarize`, but covers the full paper with better structure).
> - **`--index`** — embeds hybrid section chunks and writes a chunk index JSON.
>   Tries Ollama cloud first; if `/api/embed` is unauthorized, falls back to a
>   local Ollama daemon (run `ollama pull nomic-embed-text`).
> - **`--search`** — loads a chunk index from `--from` and prints ranked matches
>   (section, score, text snippet). Does not require a PDF argument.
> - **`--ask`** — RAG Q&A: retrieves top-k chunks, sends them as context to the
>   Ollama cloud model, and prints an answer with section citations and source
>   chunk ids. Requires `OLLAMA_API_KEY` for the chat model and a working embed
>   endpoint (same as `--index`).

You can also use the pieces directly in Python:

```python
from src.pdf_reader import read_pdf
from src.analyzer import analyze
from src.storage import save_paper, load_paper
from src.summarizer import summarize, summarize_full

full_text, page_count = read_pdf("data/AttentionIsAllYouNeed.pdf")
paper = analyze(full_text=full_text, page_count=page_count,
                source_path="data/AttentionIsAllYouNeed.pdf")

# Optional: single-pass or full-paper summarization (requires OLLAMA_API_KEY)
# paper = summarize(paper)
# paper = summarize_full(paper)

save_paper(paper, "outputs/paper.json")

paper = load_paper("outputs/paper.json")  # round-trips back into a ResearchPaper
```

### Semantic retrieval and RAG (CLI or library)

Index hybrid chunks (Abstract + numbered sections), search them, or ask
grounded questions. Chunk indexes are saved as `outputs/<pdf-stem>.chunks.json`.

```bash
python main.py data/AttentionIsAllYouNeed.pdf --index
python main.py --search "What BLEU scores were reported?" \
  --from outputs/AttentionIsAllYouNeed.chunks.json
python main.py --ask "What BLEU scores were reported?" \
  --from outputs/AttentionIsAllYouNeed.chunks.json
```

Or from Python:

```python
from src.retrieval import index_paper, search
from src.embeddings import resolve_embed_client
from src.chunk_store import save_chunk_store, default_chunk_store_path
from src.rag import ask, format_rag_answer

client = resolve_embed_client()
store = index_paper(paper, client=client)
save_chunk_store(store, default_chunk_store_path(paper))

results = search(store, "What BLEU scores were reported?", client=client, top_k=3)
answer = ask(store, "What BLEU scores were reported?", embed_client=client, top_k=5)
print(format_rag_answer(answer))
```

## Output format

The output JSON mirrors the `ResearchPaper` dataclass. Example (abridged):

```json
{
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani", "Noam Shazeer", "..."],
  "abstract": "The dominant sequence transduction models are based on ...",
  "keywords": [],
  "section_headings": ["1 Introduction", "2 Background", "3 Model Architecture", "..."],
  "page_count": 15,
  "word_count": 6166,
  "full_text": "...",
  "references": [],
  "doi": "",
  "publication_year": 0,
  "venue": "",
  "source_path": "data/AttentionIsAllYouNeed.pdf",
  "summary": {
    "tldr": "",
    "problem": "",
    "approach": "",
    "key_results": [],
    "contributions": [],
    "limitations": [],
    "key_insights": []
  }
}
```

Currently derived: `title`, `authors`, `abstract`, `keywords`, `section_headings`,
`page_count`, `word_count`, `full_text`, `source_path`. The remaining fields
(`references`, `doi`, `publication_year`, `venue`) are reserved and left at their
defaults for now. `summary` is a structured object (a "summary card") populated
by the LLM summarizer when `--summarize` or `--summarize-full` is passed;
otherwise its fields stay empty. Its fields are:

- `tldr` - a 1-2 sentence plain-English overview
- `problem` - the problem or motivation the paper addresses
- `approach` - the method or technique used
- `key_results` - main findings, including quantitative metrics when stated
- `contributions` - the paper's novel contributions
- `limitations` - caveats or weaknesses
- `key_insights` - the most important standalone takeaways

## Limitations

- Extraction is heuristic and tuned for common single-column paper layouts, so
  results may vary on other formats (e.g. multi-column or scanned PDFs).
- Field detection relies on structural cues (capitalization, numbered headings,
  superscript author markers), not semantic understanding, so it can miss or
  misidentify fields in papers that deviate from those conventions.
- Keywords are only extracted when the paper has an explicit `Keywords` /
  `Index Terms` line; otherwise the field is left empty.
- `references`, `doi`, `publication_year`, and `venue` are not yet extracted.
- `--summarize` sends only the first ~12k characters of the paper to the model,
  so very long papers may miss content from later sections. Use
  `--summarize-full` for full-paper coverage (at the cost of more API calls).
- `--summarize-full` makes one API call per chunk plus a final merge call; long
  papers take longer and cost more than `--summarize`.

## Tests

```bash
# Run everything
python -m pytest

# Unit tests only / integration only
python -m pytest tests --ignore=tests/integration
python -m pytest tests/integration
```

Unit tests are fully offline: the summarizer tests patch the Ollama network
boundary, so no API key or network access is needed. Live API integration tests
call a real Ollama cloud model and are skipped automatically unless
`OLLAMA_API_KEY` is set (e.g. via `.env`):

- `tests/integration/test_summarizer_live_api.py` — single-pass `--summarize`
- `tests/integration/test_summarizer_full_live_api.py` — chunked `--summarize-full`
- `tests/integration/test_retrieval_live_api.py` — hybrid-chunk indexing + semantic search
  (cloud embed when authorized, otherwise local `nomic-embed-text`)
- `tests/integration/test_rag_live_api.py` — RAG Q&A over indexed chunks

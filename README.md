# ResearchPaperLens

A small, modular pipeline that reads a research-paper PDF, extracts structured
metadata (title, authors, abstract, section headings, and more), and saves the
result as clean JSON.

## How it works

The pipeline is split into focused, independently testable modules:

```
.
├── src/
│   ├── paper.py        # ResearchPaper data model (dataclass)
│   ├── pdf_reader.py   # PDF I/O: extract full text + page count
│   ├── analyzer.py     # Pure analysis: text -> ResearchPaper fields
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
```

`analyzer.py` is deliberately decoupled from PDF reading: it only accepts
already-extracted text, which keeps the analysis logic easy to test and reusable
for any text source.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the pipeline on a PDF:

```bash
# Use the bundled sample paper -> outputs/AttentionIsAllYouNeed.json
python main.py

# Analyze your own PDF (output defaults to outputs/<pdf-stem>.json)
python main.py path/to/paper.pdf

# Choose an explicit output file
python main.py path/to/paper.pdf -o results/paper.json
```

See all options with `python main.py --help`.

You can also use the pieces directly in Python:

```python
from src.pdf_reader import read_pdf
from src.analyzer import analyze
from src.storage import save_paper, load_paper

full_text, page_count = read_pdf("data/AttentionIsAllYouNeed.pdf")
paper = analyze(full_text=full_text, page_count=page_count,
                source_path="data/AttentionIsAllYouNeed.pdf")
save_paper(paper, "outputs/paper.json")

paper = load_paper("outputs/paper.json")  # round-trips back into a ResearchPaper
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
  "source_path": "data/AttentionIsAllYouNeed.pdf"
}
```

Currently derived: `title`, `authors`, `abstract`, `keywords`, `section_headings`,
`page_count`, `word_count`, `full_text`, `source_path`. The remaining fields
(`references`, `doi`, `publication_year`, `venue`) are reserved and left at their
defaults for now.

> Extraction is heuristic and tuned for common single-column paper layouts, so
> results may vary on other formats.

## Tests

```bash
# Run everything
python -m pytest

# Unit tests only / integration only
python -m pytest tests --ignore=tests/integration
python -m pytest tests/integration
```

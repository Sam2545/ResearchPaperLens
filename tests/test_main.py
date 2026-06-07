"""Tests for the CLI argument parsing and output-path handling in ``main``."""

from __future__ import annotations

from pathlib import Path

import pytest

import main as main_module
from main import DEFAULT_PDF, format_search_results, parse_args, process_pdf, run_search
from src.chunk_store import IndexedChunk
from src.storage import load_paper
from src.summarizer import DEFAULT_MODEL
from src.vector_index import SearchResult

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "AttentionIsAllYouNeed.pdf"


def test_parse_args_defaults():
    args = parse_args([])
    assert args.pdf is None
    assert args._default_pdf == str(DEFAULT_PDF)
    assert args.output is None


def test_parse_args_positional_pdf():
    args = parse_args(["some/paper.pdf"])
    assert args.pdf == "some/paper.pdf"
    assert args.output is None


def test_parse_args_output_flag():
    args = parse_args(["some/paper.pdf", "-o", "out/result.json"])
    assert args.pdf == "some/paper.pdf"
    assert args.output == "out/result.json"

    args_long = parse_args(["some/paper.pdf", "--output", "out/result.json"])
    assert args_long.output == "out/result.json"


def test_process_pdf_uses_explicit_output_path(tmp_path):
    target = tmp_path / "custom_name.json"
    result, chunk_path = process_pdf(PDF_PATH, output_path=target)
    assert result == target
    assert chunk_path is None
    assert target.is_file()
    assert load_paper(target).title == "Attention Is All You Need"


def test_process_pdf_defaults_to_output_dir(tmp_path):
    result, chunk_path = process_pdf(PDF_PATH, output_dir=tmp_path)
    assert result == tmp_path / "AttentionIsAllYouNeed.json"
    assert chunk_path is None
    assert result.is_file()


def test_parse_args_summarize_defaults_false():
    args = parse_args([])
    assert args.summarize is False


def test_parse_args_summarize_flag():
    args = parse_args(["some/paper.pdf", "--summarize"])
    assert args.summarize is True


def test_parse_args_model_default():
    args = parse_args([])
    assert args.model == DEFAULT_MODEL


def test_parse_args_model_flag():
    args = parse_args(["some/paper.pdf", "--summarize", "--model", "gpt-oss:20b"])
    assert args.model == "gpt-oss:20b"


def test_process_pdf_summarize_runs_summarizer(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_summarize(paper, *, model):
        calls.append(paper.title)
        return paper

    monkeypatch.setattr(main_module, "summarize", fake_summarize)
    process_pdf(PDF_PATH, output_dir=tmp_path, summarize_paper=True)
    assert calls == ["Attention Is All You Need"]


def test_process_pdf_forwards_model_to_summarizer(tmp_path, monkeypatch):
    used_models: list[str] = []

    def fake_summarize(paper, *, model):
        used_models.append(model)
        return paper

    monkeypatch.setattr(main_module, "summarize", fake_summarize)
    process_pdf(
        PDF_PATH, output_dir=tmp_path, summarize_paper=True, model="gpt-oss:20b"
    )
    assert used_models == ["gpt-oss:20b"]


def test_process_pdf_skips_summarizer_by_default(tmp_path, monkeypatch):
    def fail_summarize(paper, *, model):
        raise AssertionError("summarizer should not run without summarize_paper")

    monkeypatch.setattr(main_module, "summarize", fail_summarize)
    monkeypatch.setattr(main_module, "summarize_full", fail_summarize)
    process_pdf(PDF_PATH, output_dir=tmp_path)


def test_parse_args_summarize_full_defaults_false():
    args = parse_args([])
    assert args.summarize_full is False


def test_parse_args_summarize_full_flag():
    args = parse_args(["some/paper.pdf", "--summarize-full"])
    assert args.summarize_full is True
    assert args.summarize is False


def test_parse_args_summarize_and_summarize_full_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["some/paper.pdf", "--summarize", "--summarize-full"])


def test_process_pdf_summarize_full_runs_full_summarizer(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_summarize_full(paper, *, model, **kwargs):
        calls.append(paper.title)
        return paper

    monkeypatch.setattr(main_module, "summarize_full", fake_summarize_full)
    process_pdf(PDF_PATH, output_dir=tmp_path, summarize_full_paper=True)
    assert calls == ["Attention Is All You Need"]


def test_process_pdf_forwards_model_to_summarize_full(tmp_path, monkeypatch):
    used_models: list[str] = []

    def fake_summarize_full(paper, *, model, **kwargs):
        used_models.append(model)
        return paper

    monkeypatch.setattr(main_module, "summarize_full", fake_summarize_full)
    process_pdf(
        PDF_PATH,
        output_dir=tmp_path,
        summarize_full_paper=True,
        model="gpt-oss:20b",
    )
    assert used_models == ["gpt-oss:20b"]


def test_parse_args_index_flag():
    args = parse_args(["some/paper.pdf", "--index"])
    assert args.index is True


def test_parse_args_search_and_from():
    args = parse_args(["--search", "BLEU score", "--from", "outputs/paper.chunks.json"])
    assert args.search == "BLEU score"
    assert args.chunk_store == "outputs/paper.chunks.json"
    assert args.pdf is None


def test_parse_args_top_k_default():
    args = parse_args(["--search", "query", "--from", "index.json"])
    assert args.top_k == 5


def test_parse_args_top_k_flag():
    args = parse_args(["--search", "query", "--from", "index.json", "--top-k", "3"])
    assert args.top_k == 3


def test_main_search_requires_from(capsys):
    with pytest.raises(SystemExit) as exc:
        main_module.main(["--search", "BLEU"])
    assert exc.value.code == 2
    assert "--from" in capsys.readouterr().err


def test_process_pdf_index_saves_chunk_store(tmp_path, monkeypatch):
    fake_client = object()

    def fake_index_paper(paper, *, client=None, model="nomic-embed-text"):
        assert client is fake_client
        from src.chunk_store import ChunkStore

        return ChunkStore(
            paper_source=paper.source_path,
            paper_title=paper.title,
            embed_model=model,
            embed_dimensions=2,
            chunks=[],
        )

    monkeypatch.setattr(main_module, "resolve_embed_client", lambda: fake_client)
    monkeypatch.setattr(main_module, "index_paper", fake_index_paper)

    _, chunk_path = process_pdf(
        PDF_PATH,
        output_dir=tmp_path,
        index_chunks=True,
        embed_client=fake_client,
    )
    assert chunk_path == tmp_path / "AttentionIsAllYouNeed.chunks.json"
    assert chunk_path.is_file()


def test_run_search_delegates_to_retrieval(tmp_path, monkeypatch):
    chunk_path = tmp_path / "paper.chunks.json"
    chunk_path.write_text("{}", encoding="utf-8")
    calls: list[tuple] = []

    def fake_search(store, query, *, client=None, model=None, top_k=5):
        calls.append((query, top_k, client))
        chunk = IndexedChunk(
            id="a",
            paper_source="data/paper.pdf",
            section_heading="6 Results",
            text="BLEU 28.4",
            word_count=2,
            start_word=0,
            end_word=2,
            embedding=[1.0, 0.0],
        )
        return [SearchResult(chunk=chunk, score=0.95)]

    fake_client = object()
    monkeypatch.setattr(main_module, "load_chunk_store", lambda path: object())
    monkeypatch.setattr(main_module, "search", fake_search)

    results = run_search(chunk_path, "BLEU score", top_k=2, embed_client=fake_client)
    assert len(results) == 1
    assert calls == [("BLEU score", 2, fake_client)]


def test_format_search_results_includes_section_and_score():
    chunk = IndexedChunk(
        id="paper:6-results:0",
        paper_source="data/paper.pdf",
        section_heading="6 Results",
        text="Our model achieves BLEU scores of 28.4 on WMT.",
        word_count=9,
        start_word=0,
        end_word=9,
        embedding=[1.0, 0.0],
    )
    text = format_search_results(
        [SearchResult(chunk=chunk, score=0.9123)],
        query="BLEU score",
        chunk_store_path="outputs/paper.chunks.json",
    )
    assert "BLEU score" in text
    assert "6 Results" in text
    assert "0.9123" in text
    assert "paper:6-results:0" in text

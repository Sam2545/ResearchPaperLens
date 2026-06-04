"""Tests for the CLI argument parsing and output-path handling in ``main``."""

from __future__ import annotations

from pathlib import Path

import main as main_module
from main import DEFAULT_PDF, parse_args, process_pdf
from src.storage import load_paper

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "AttentionIsAllYouNeed.pdf"


def test_parse_args_defaults():
    args = parse_args([])
    assert args.pdf == str(DEFAULT_PDF)
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
    result = process_pdf(PDF_PATH, output_path=target)
    assert result == target
    assert target.is_file()
    assert load_paper(target).title == "Attention Is All You Need"


def test_process_pdf_defaults_to_output_dir(tmp_path):
    result = process_pdf(PDF_PATH, output_dir=tmp_path)
    assert result == tmp_path / "AttentionIsAllYouNeed.json"
    assert result.is_file()


def test_parse_args_summarize_defaults_false():
    args = parse_args([])
    assert args.summarize is False


def test_parse_args_summarize_flag():
    args = parse_args(["some/paper.pdf", "--summarize"])
    assert args.summarize is True


def test_process_pdf_summarize_runs_summarizer(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_summarize(paper):
        calls.append(paper.title)
        return paper

    monkeypatch.setattr(main_module, "summarize", fake_summarize)
    process_pdf(PDF_PATH, output_dir=tmp_path, summarize_paper=True)
    assert calls == ["Attention Is All You Need"]


def test_process_pdf_skips_summarizer_by_default(tmp_path, monkeypatch):
    def fail_summarize(paper):
        raise AssertionError("summarizer should not run without summarize_paper")

    monkeypatch.setattr(main_module, "summarize", fail_summarize)
    process_pdf(PDF_PATH, output_dir=tmp_path)

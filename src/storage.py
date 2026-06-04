"""Persistence helpers for loading and saving paper data."""


def save(data: dict, path: str) -> None:
    """Save analysis results to disk."""
    raise NotImplementedError


def load(path: str) -> dict:
    """Load analysis results from disk."""
    raise NotImplementedError

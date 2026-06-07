"""Shared Ollama cloud client construction for API-backed modules."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from ollama import Client

OLLAMA_CLOUD_HOST = "https://ollama.com"
API_KEY_ENV = "OLLAMA_API_KEY"


def build_cloud_client(
    *,
    host: str = OLLAMA_CLOUD_HOST,
    api_key: str | None = None,
) -> Client:
    """Build an Ollama client pointed at the cloud API.

    The API key is read from ``api_key`` or, if omitted, ``OLLAMA_API_KEY``
    (loaded from a local ``.env`` file when present). Raises
    :class:`RuntimeError` when no key is available.
    """
    if api_key is None:
        load_dotenv()
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Create an API key at "
            "https://ollama.com/settings/keys and export it, e.g. "
            f"`export {API_KEY_ENV}=your_api_key`."
        )
    return Client(host=host, headers={"Authorization": f"Bearer {key}"})

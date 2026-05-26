"""Baseline agent — one direct Gemini call per image. The floor to beat."""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

NAME = "baseline"
DESCRIPTION = "Single direct gemini-2.5-flash call — no tools, no orchestration."
MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None


def _client_lazy() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4"),
        )
    return _client


PROMPT = (
    "You will be shown an image and asked how many of something are in it. "
    "Count as precisely as you can. Reply with ONLY an integer on the first line."
)


def _parse_int(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


async def answer(image_bytes: bytes, mime_type: str, question: str) -> int:
    resp = await _client_lazy().aio.models.generate_content(
        model=MODEL,
        contents=[
            f"{PROMPT}\n\nQuestion: {question}",
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    n = _parse_int(resp.text or "")
    if n is None:
        raise ValueError(f"could not parse integer from: {resp.text!r}")
    return n

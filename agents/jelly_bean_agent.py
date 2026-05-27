"""Jelly Bean agent scaffold for dense object counting."""
from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import re
import statistics

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv(override=True)

NAME = "jelly-bean-agent"
DESCRIPTION = "Uses a divide-and-conquer spatial grid and ensemble averaging to accurately count dense objects."
MODEL = "gemini-2.5-flash"

_logger = logging.getLogger(__name__)
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


CHUNK_COUNTER_INSTRUCTION = "\n\n".join(
    [
        "You are a precise counting tool. Analyze this image grid chunk.",
        "Boundary Rule: To avoid double-counting across chunks, you must count all whole jelly beans, PLUS any partial jelly beans touching the TOP and LEFT borders of the image. Do NOT count partial jelly beans touching the BOTTOM or RIGHT borders.",
        "Output Format: First, provide a brief reasoning scratchpad where you scan the image and tally the beans. Then, on a new line at the very end, output strictly: FINAL ANSWER: <integer>.",
    ]
)


def split_image_into_grid(image_bytes: bytes, rows: int = 2, cols: int = 2) -> list[bytes]:
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    chunk_width = width / cols
    chunk_height = height / rows
    chunks: list[bytes] = []

    for row in range(rows):
        for col in range(cols):
            left = round(col * chunk_width)
            upper = round(row * chunk_height)
            right = round((col + 1) * chunk_width)
            lower = round((row + 1) * chunk_height)
            cropped = image.crop((left, upper, right, lower))

            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            chunks.append(buffer.getvalue())

    return chunks


async def count_single_chunk(chunk_bytes: bytes, sem: asyncio.Semaphore, mime_type: str = "image/png") -> int:
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            async with sem:
                resp = await _client_lazy().aio.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_bytes(data=chunk_bytes, mime_type=mime_type),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=CHUNK_COUNTER_INSTRUCTION,
                    ),
                )
            
            lines = [line.strip() for line in (resp.text or "").strip().splitlines() if line.strip()]
            if lines:
                final_line = lines[-1].replace("*", "").replace(".", "")
                match = re.search(r"(\d+)", final_line)
                if match:
                    return int(match.group(1))
            
            _logger.warning("Chunk counter did not output a valid integer: %r", resp.text)
            print(f"DEBUG: Parse Failure for raw response.text:\n{resp.text}")
            return 0
                
        except Exception as e:
            _logger.warning("Attempt %d failed for chunk counter: %r", attempt + 1, e)
            if attempt < max_attempts - 1:
                # Exponential backoff with jitter: 2s, 4s... plus random jitter
                delay = (2 ** (attempt + 1)) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                _logger.error("All %d attempts failed for chunk counter.", max_attempts)
                if 'resp' in locals():
                    print(f"DEBUG: Exception Parse Failure for raw response.text:\n{resp.text}")
                
    # Fallback if all passes fail
    return 0


async def get_chunk_consensus(chunk_bytes: bytes, sem: asyncio.Semaphore, passes: int = 3) -> int:
    counts = await asyncio.gather(*[count_single_chunk(chunk_bytes, sem) for _ in range(passes)])

    valid_counts = [c for c in counts if c > 0]
    if not valid_counts:
        return 0

    return int(statistics.median(valid_counts))


async def answer(image_bytes: bytes, mime_type: str, question: str) -> int:
    q_lower = question.lower()
    sem = asyncio.Semaphore(5)

    if any(keyword in q_lower for keyword in ["triangle", "square", "figure", "puzzle"]):
        _logger.info("Geometric puzzle detected. Bypassing grid splitter.")
        total_count = await get_chunk_consensus(image_bytes, sem)
        _logger.info("Total geometric count: %d", total_count)
        return total_count

    chunks = split_image_into_grid(image_bytes, rows=2, cols=2)
    _logger.info("Generated %d image chunks for jelly bean counting.", len(chunks))

    chunk_tasks = [get_chunk_consensus(chunk_bytes, sem) for chunk_bytes in chunks]
    chunk_counts = await asyncio.gather(*chunk_tasks)

    total_jelly_beans = 0
    for index, count in enumerate(chunk_counts, start=1):
        _logger.info("Chunk %d consensus count: %d", index, count)
        total_jelly_beans += count

    _logger.info("Total jelly bean count: %d", total_jelly_beans)
    return int(total_jelly_beans)

"""Vision-router counting agent for diverse image counting tasks."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import random
import re
import statistics
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv(override=True)

NAME = "jelly-bean-agent"
DESCRIPTION = "Routes each image through a scene analyzer before choosing full-image or grid-based counting."
MODEL = "gemini-2.5-flash"

_logger = logging.getLogger(__name__)
_client: genai.Client | None = None
_counter_sem: asyncio.Semaphore | None = None


def _client_lazy() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4"),
        )
    return _client


def _counter_semaphore() -> asyncio.Semaphore:
    global _counter_sem
    if _counter_sem is None:
        _counter_sem = asyncio.Semaphore(5)
    return _counter_sem


def typical_item_size_cm(item_name: str) -> dict[str, float]:
    """Return the typical diameter (in cm) of a common small item.

    Use when the image is a jar/container of small items and you want to
    estimate the count by volume rather than counting one-by-one.

    Args:
        item_name: e.g. "jelly bean", "marble", "coin".
    """
    table = {
        "jelly bean": 2.0, "jellybean": 2.0,
        "m&m": 1.3, "skittle": 1.3, "gumball": 2.5,
        "marble": 1.6, "coin": 2.3, "penny": 1.9,
        "apple": 7.5, "orange": 7.0, "egg": 5.5,
    }
    key = item_name.strip().lower()
    return {"diameter_cm": table.get(key, 1.5), "known": key in table}


def volume_estimator(
    container_shape: str,
    container_dims_cm: list[float],
    item_avg_size_cm: float,
    packing_efficiency: float = 0.64,
) -> dict[str, Any]:
    """Estimate how many items fit in a container by volume.

    Args:
        container_shape: 'cylinder', 'sphere', or 'box'.
        container_dims_cm: cylinder=[radius, height], sphere=[radius],
            box=[width, height, depth].
        item_avg_size_cm: approximate diameter of one item in cm.
        packing_efficiency: fraction of volume filled. 0.64 = random
            close-packed spheres.
    """
    s = container_shape.lower()
    if s == "cylinder" and len(container_dims_cm) >= 2:
        r, h = container_dims_cm[:2]
        v = math.pi * r * r * h
    elif s == "sphere" and len(container_dims_cm) >= 1:
        v = 4 / 3 * math.pi * container_dims_cm[0] ** 3
    elif s == "box" and len(container_dims_cm) >= 3:
        w, h, d = container_dims_cm[:3]
        v = w * h * d
    else:
        return {"estimate": 0, "notes": "invalid shape or dims"}
    v_item = 4 / 3 * math.pi * (item_avg_size_cm / 2) ** 3
    return {"estimate": int(v * packing_efficiency / v_item)}


def _shrink(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    im = Image.open(io.BytesIO(image_bytes))
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    fmt = "JPEG" if mime.endswith("jpeg") or mime.endswith("jpg") else "PNG"
    im.convert("RGB" if fmt == "JPEG" else "RGBA").save(buf, format=fmt, quality=90)
    return buf.getvalue(), f"image/{fmt.lower()}"


SCENE_ANALYZER_INSTRUCTION = """Analyze this image. Categorize the scene to determine the best counting strategy.

Return only valid JSON with this exact shape:
{
  "category": "geometric_puzzle" | "scattered_objects" | "mixed_classes",
  "spatial_layout": "flat_spread" | "3d_container",
  "density": "low" | "high",
  "distractor_warning": "brief downstream rule, or empty string",
  "recommended_grid": 1 | 2 | 3
}

Definitions:
- category="geometric_puzzle" for line/shape combinatorics such as counting all triangles or all squares, including compound shapes. These must keep the full image intact.
- category="mixed_classes" when the image contains multiple object classes, colors, text styles, or other likely distractors.
- category="scattered_objects" when the target objects are physical items spread across the scene without major distractor classes.
- spatial_layout must be strictly either "flat_spread" or "3d_container". Determine if the items are scattered on a flat surface where most items are visible, or if they are piled inside a 3D container (like a jar or bowl) where inner items are hidden from view.
- density="low" for fewer than 15 target items or sparse layouts; density="high" for dense clusters, jars, piles, or many repeated objects.
- recommended_grid=1 for geometric puzzles, low-density scenes, or any case where cropping would damage the evidence.
- recommended_grid=2 for moderately dense scattered objects.
- recommended_grid=3 for dense clusters where spatial chunking helps.

The distractor_warning should be an imperative rule for the counter, for example:
"Ignore the cats, count only dogs" or "Count only blue numbers". Leave it empty when there is no distractor."""

SCENE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["geometric_puzzle", "scattered_objects", "mixed_classes"],
        },
        "spatial_layout": {"type": "string", "enum": ["flat_spread", "3d_container"]},
        "density": {"type": "string", "enum": ["low", "high"]},
        "distractor_warning": {"type": "string"},
        "recommended_grid": {"type": "integer"},
    },
    "required": [
        "category",
        "spatial_layout",
        "density",
        "distractor_warning",
        "recommended_grid",
    ],
}

CONTAINER_COUNTING_OVERRIDE = (
    "CRITICAL: This is a 3D container. You cannot count these one-by-one because the inner items are hidden. "
    "You MUST call the typical_item_size_cm and volume_estimator tools to calculate the final count mathematically. "
    "Output your reasoning, then FINAL ANSWER: <integer>."
)

ALLOWED_CATEGORIES = {"geometric_puzzle", "scattered_objects", "mixed_classes"}
ALLOWED_SPATIAL_LAYOUTS = {"flat_spread", "3d_container"}
ALLOWED_DENSITIES = {"low", "high"}


def _clamp_grid(value: object) -> int:
    try:
        grid = int(value)
    except (TypeError, ValueError):
        return 2
    return min(3, max(1, grid))


def _fallback_scene_analysis() -> dict[str, object]:
    return {
        "category": "scattered_objects",
        "spatial_layout": "flat_spread",
        "density": "low",
        "distractor_warning": "",
        "recommended_grid": 1,
    }


def _normalise_scene_analysis(raw: dict[str, object]) -> dict[str, object]:
    fallback = _fallback_scene_analysis()
    category = raw.get("category")
    spatial_layout = raw.get("spatial_layout")
    density = raw.get("density")
    distractor_warning = raw.get("distractor_warning")

    if category not in ALLOWED_CATEGORIES:
        category = fallback["category"]
    if spatial_layout not in ALLOWED_SPATIAL_LAYOUTS:
        spatial_layout = fallback["spatial_layout"]
    if density not in ALLOWED_DENSITIES:
        density = fallback["density"]

    grid = _clamp_grid(raw.get("recommended_grid", fallback["recommended_grid"]))
    if category == "geometric_puzzle" or spatial_layout == "3d_container":
        grid = 1

    return {
        "category": str(category),
        "spatial_layout": str(spatial_layout),
        "density": str(density),
        "distractor_warning": str(distractor_warning or ""),
        "recommended_grid": grid,
    }


def _parse_json_object(text: str) -> dict[str, object] | None:
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


async def analyze_scene(image_bytes: bytes, mime_type: str = "image/png") -> dict[str, object]:
    """SceneAnalyzer agent: choose the routing strategy, but do not count."""
    try:
        resp = await _client_lazy().aio.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SCENE_ANALYZER_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SCENE_ANALYSIS_SCHEMA,
                temperature=0,
            ),
        )
        parsed = _parse_json_object(resp.text or "")
        if parsed is None:
            raise ValueError(f"scene analyzer returned non-JSON: {resp.text!r}")
        strategy = _normalise_scene_analysis(parsed)
        _logger.info("SceneAnalyzer strategy: %s", strategy)
        return strategy
    except Exception as e:
        fallback = _fallback_scene_analysis()
        _logger.warning("SceneAnalyzer failed, using fallback %s: %r", fallback, e)
        return fallback


def build_counter_instruction(
    distractor_warning: str,
    *,
    is_grid_chunk: bool,
    density: str,
    absolute_override: str | None = None,
) -> str:
    scope_rule = (
        "Scope: You are analyzing one cropped grid chunk from the original image. "
        "Use local positions inside this crop, such as top-left, center, or lower-right of the chunk. "
        "Boundary rule: count all whole target items, plus target items crossing the TOP or LEFT crop borders. "
        "Do NOT count target items crossing the BOTTOM or RIGHT crop borders."
        if is_grid_chunk
        else "Scope: You are analyzing the entire original image. Keep the full spatial layout intact, especially for geometric puzzles where cropping destroys the evidence."
    )

    distractor_rule = (
        f"CRITICAL RULE: {distractor_warning.strip()}"
        if distractor_warning.strip()
        else "CRITICAL RULE: Count only the target described by the orchestrator."
    )
    if absolute_override:
        density_rule = absolute_override.strip()
    elif density == "high" and not is_grid_chunk:
        density_rule = (
            "CRITICAL STRATEGY: This is a high-density whole-image case. If the scene is a jar, container, pile, or other volume-estimation setup, "
            "call typical_item_size_cm and volume_estimator to calculate the final count mathematically. If reliable container dimensions cannot be inferred "
            "from the image, do not refuse; use spatial clustering, density, and visible exemplars to estimate. Always output reasoning, then FINAL ANSWER: <integer>."
        )
    elif density == "high":
        density_rule = (
            "Strategy: This is a high-density crop. Do not use volume_estimator on a crop because global container dimensions are unavailable. "
            "Map local clusters or rows, estimate dense groups if needed, apply the crop boundary rule, and always output FINAL ANSWER: <integer>."
        )
    else:
        density_rule = "Strategy: This is a low-density or sparse image. Count directly using the spatial map."

    instruction_parts = [
        "You are the GroundedCounter worker in a multi-agent vision-router counting pipeline.",
        "Single responsibility: count the target you are instructed to count. Do not decide routing, crop strategy, or scene category.",
        distractor_rule,
        density_rule,
        scope_rule,
        "If the target rule filters by class, color, text, fullness, or any other attribute, apply that filter strictly before counting.",
    ]
    if absolute_override:
        instruction_parts.extend(
            [
                "Use the full-image container geometry, visible fill level, and typical item size to estimate the hidden volume mathematically.",
                "Output format: Provide your reasoning and tool-based calculation, then on a new line at the very end output strictly: FINAL ANSWER: <integer>.",
            ]
        )
    else:
        instruction_parts.extend(
            [
                "For geometric puzzles, preserve the whole diagram mentally and count requested line-defined or compound shapes across all sizes.",
                "For object scenes, count visible target instances; if objects overlap, use contours and local grouping to avoid double-counting.",
                "Spatial grounding requirement: In your reasoning scratchpad, do not just tally numbers. First verbally map the image region by rough location, for example: 'Top left: 1 coin. Center: 3 overlapping coins. Lower right: 2 coins.' For dense scenes, group by local clusters or rows and provide subtotals. For geometric puzzles, map relevant regions and shape sizes before summing.",
                "Output format: Provide the spatial map scratchpad and any subtotals, then on a new line at the very end output strictly: FINAL ANSWER: <integer>.",
            ]
        )

    return "\n\n".join(instruction_parts)


def _parse_count(text: str) -> int | None:
    if not text:
        return None

    final_match = re.search(r"FINAL ANSWER:\s*(-?\d[\d,]*)", text, re.IGNORECASE)
    if final_match:
        return int(final_match.group(1).replace(",", ""))

    numbers = re.findall(r"-?\d[\d,]*", text.replace(",", ""))
    return int(numbers[-1]) if numbers else None


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


async def count_image_target(
    image_bytes: bytes,
    distractor_warning: str,
    is_grid_chunk: bool,
    density: str = "low",
    mime_type: str = "image/png",
    absolute_override: str | None = None,
) -> int:
    """GroundedCounter agent: count only the instructed target."""
    max_attempts = 3
    worker_name = "ChunkCounter" if is_grid_chunk else "WholeImageCounter"
    instruction = build_counter_instruction(
        distractor_warning,
        is_grid_chunk=is_grid_chunk,
        density=density,
        absolute_override=absolute_override,
    )

    for attempt in range(max_attempts):
        try:
            async with _counter_semaphore():
                resp = await _client_lazy().aio.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=instruction,
                        tools=[typical_item_size_cm, volume_estimator],
                        temperature=0,
                    ),
                )

            count = _parse_count(resp.text or "")
            if count is not None:
                return count

            _logger.warning("%s did not output a valid integer: %r", worker_name, resp.text)
            print(f"DEBUG: Parse Failure for raw response.text:\n{resp.text}")

        except Exception as e:
            _logger.warning("Attempt %d failed for %s: %r", attempt + 1, worker_name, e)

        if attempt < max_attempts - 1:
            # Exponential backoff with jitter: 2s, 4s... plus random jitter
            delay = (2 ** (attempt + 1)) + random.uniform(0, 1)
            await asyncio.sleep(delay)
        else:
            _logger.error("All %d attempts failed for %s.", max_attempts, worker_name)
            if 'resp' in locals():
                print(f"DEBUG: Exception Parse Failure for raw response.text:\n{resp.text}")

    # Fallback if all passes fail
    return 0


async def run_consensus_manager(
    image_bytes: bytes,
    distractor_warning: str,
    is_grid_chunk: bool = False,
    density: str = "low",
    mime_type: str = "image/png",
    passes: int = 3,
    absolute_override: str | None = None,
) -> int:
    counts = await asyncio.gather(
        *[
            count_image_target(
                image_bytes,
                distractor_warning,
                is_grid_chunk=is_grid_chunk,
                density=density,
                mime_type=mime_type,
                absolute_override=absolute_override,
            )
            for _ in range(passes)
        ]
    )

    valid_counts = [c for c in counts if c >= 0]
    if not valid_counts:
        return 0

    consensus_count = int(statistics.median(valid_counts))
    _logger.info("ConsensusManager counts=%s median=%d", counts, consensus_count)
    return consensus_count


def _build_grounded_counter_rule(question: str, distractor_warning: object) -> str:
    """Give the worker its target without making it reason about routing."""
    rules = [f"Answer this counting question exactly: {question}"]
    warning = str(distractor_warning or "").strip()
    if warning:
        rules.append(warning)
    return " ".join(rules)


async def answer(image_bytes: bytes, mime_type: str, question: str) -> int:
    image_bytes, mime_type = _shrink(image_bytes, mime_type)
    strategy = await analyze_scene(image_bytes, mime_type)
    category = str(strategy.get("category", "scattered_objects"))
    spatial_layout = str(strategy.get("spatial_layout", "flat_spread"))
    density = str(strategy.get("density", "low"))
    grid_size = _clamp_grid(strategy.get("recommended_grid", 2))
    counter_rule = _build_grounded_counter_rule(question, strategy.get("distractor_warning", ""))

    async def run_full_image_consensus(reason: str) -> int:
        _logger.info("Path A selected: %s. Sending whole image to consensus manager.", reason)
        total_count = await run_consensus_manager(
            image_bytes,
            counter_rule,
            is_grid_chunk=False,
            density=density,
            mime_type=mime_type,
        )
        _logger.info("Total full-image count: %d", total_count)
        return total_count

    if spatial_layout == "3d_container":
        _logger.info(
            "Path C selected: spatial_layout=3d_container. Sending whole image to GroundedCounter with volume override."
        )
        total_count = await count_image_target(
            image_bytes,
            counter_rule,
            is_grid_chunk=False,
            density=density,
            mime_type=mime_type,
            absolute_override=CONTAINER_COUNTING_OVERRIDE,
        )
        _logger.info("Total 3D-container count: %d", total_count)
        return total_count

    if category == "geometric_puzzle" or grid_size == 1:
        return await run_full_image_consensus(
            f"category={category}, recommended_grid={grid_size}"
        )

    chunks = split_image_into_grid(image_bytes, rows=grid_size, cols=grid_size)
    _logger.info(
        "Path B selected: category=%s with %dx%d grid; generated %d chunks.",
        category,
        grid_size,
        grid_size,
        len(chunks),
    )

    chunk_tasks = [
        run_consensus_manager(
            chunk_bytes,
            counter_rule,
            is_grid_chunk=True,
            density="low",
            mime_type="image/png",
        )
        for chunk_bytes in chunks
    ]
    chunk_counts = await asyncio.gather(*chunk_tasks)

    total_count = 0
    for index, count in enumerate(chunk_counts, start=1):
        _logger.info("Chunk %d consensus count: %d", index, count)
        total_count += count

    _logger.info("Total routed grid count: %d", total_count)
    return int(total_count)

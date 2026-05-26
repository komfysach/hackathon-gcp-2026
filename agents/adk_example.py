"""ADK example agent — small, runnable, lots of `# EXTEND:` hints.

Read it top-to-bottom once, then look at `# EXTEND:` comments — they mark the
places where you can plug in something smarter to beat the baseline.
"""
from __future__ import annotations

import io
import math
import os
import re
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from PIL import Image

load_dotenv(override=True)

NAME = "adk_example"
DESCRIPTION = "ADK agent with two tools (item-size lookup + volume estimator)."
MODEL = "gemini-2.5-flash"
# EXTEND: Try `gemini-2.5-pro` for harder images (slower but more accurate).


# ---------------------------------------------------------------------------
# Tools — any plain Python function with type hints + docstring becomes a tool.
# The docstring is what the model sees — write it like you would write to a
# junior engineer.
# ---------------------------------------------------------------------------


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


# EXTEND: Add your own tools. Ideas:
#
#   def grid_overlay_advice(rows: int, cols: int) -> str:
#       """Tell the agent to mentally split the image into a rows×cols grid,
#       count each cell separately, and sum. Helps for 50-500 items."""
#
# You can also use first-party ADK tools — `from google.adk.tools import
# google_search` lets the agent look up "how many eggs in a standard carton".


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------

INSTRUCTION = """You are a precise counting agent. You'll be shown an image and asked how many of something are in it.

Strategy:
  - Small count (< 50, every item visible): count directly.
  - Medium count (50–500): mentally split the image into a 3×3 grid, count each cell, sum.
  - Large count (> 500, e.g. jelly bean jars): call typical_item_size_cm then volume_estimator.

End your reply with `FINAL ANSWER: <integer>` on its own line. Be decisive — commit to one integer."""

# EXTEND: A stronger prompt is one of the biggest wins. Try:
#   - Few-shot examples ("here's a 12-egg carton, the answer is 12").
#   - Force the model to verbalise per-row counts before summing.
#   - Tell it to double-check by a second method and reconcile.

root_agent = Agent(
    name="counting_agent",
    model=MODEL,
    description="Counts items in an image.",
    instruction=INSTRUCTION,
    tools=[typical_item_size_cm, volume_estimator],
)

# EXTEND: Multi-agent ensemble — two counters with different strategies run in
# parallel, then a reconciler picks the best answer. Uncomment the block below
# to replace the single agent with the ensemble.
#
# from google.adk.agents import SequentialAgent, ParallelAgent
#
# counter_a = Agent(
#     name="direct_counter",
#     model=MODEL,
#     description="Counts items directly, one by one.",
#     instruction="""You are a precise counting agent. You will be shown an image and a question about how many of something are in it.
#
# Your strategy: count every item directly and systematically.
#   - Scan left-to-right, top-to-bottom.
#   - Number each item as you go (1, 2, 3, ...).
#   - For larger counts, split into rows or groups and subtotal each.
#
# End your reply with `ESTIMATE: <integer>` on its own line.""",
# )
#
# counter_b = Agent(
#     name="grid_counter",
#     model=MODEL,
#     description="Counts items by dividing the image into a grid.",
#     instruction="""You are a precise counting agent. You will be shown an image and a question about how many of something are in it.
#
# Your strategy: divide and conquer using a mental grid.
#   - Split the image into a 3×3 grid (9 cells).
#   - Count the items in each cell separately, noting partial items on borders.
#   - Sum the 9 cell counts for your total.
#   - For very large counts (jars, piles), estimate density × area instead.
#
# End your reply with `ESTIMATE: <integer>` on its own line.""",
#     tools=[typical_item_size_cm, volume_estimator],
# )
#
# reconciler = Agent(
#     name="reconciler",
#     model=MODEL,
#     description="Reconciles two counting estimates into a final answer.",
#     instruction="""You are a reconciliation agent. Two counting agents have each independently estimated how many items are in an image. Their estimates appear earlier in the conversation.
#
# Your job:
#   1. Find both ESTIMATE values from the earlier messages.
#   2. If they agree (within 10%), use their average.
#   3. If they disagree significantly, reason about which strategy was more appropriate for this image and weight that estimate more heavily.
#   4. Round to the nearest integer.
#
# End your reply with `FINAL ANSWER: <integer>` on its own line.""",
# )
#
# root_agent = SequentialAgent(
#     name="ensemble",
#     description="Ensemble of two counters + a reconciler.",
#     sub_agents=[
#         ParallelAgent(name="counters", sub_agents=[counter_a, counter_b]),
#         reconciler,
#     ],
# )

runner = InMemoryRunner(agent=root_agent, app_name="ggg_kit")


# ---------------------------------------------------------------------------
# Image preprocessing — keep payloads small so calls stay fast.
# ---------------------------------------------------------------------------


def _shrink(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    im = Image.open(io.BytesIO(image_bytes))
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    fmt = "JPEG" if mime.endswith("jpeg") or mime.endswith("jpg") else "PNG"
    im.convert("RGB" if fmt == "JPEG" else "RGBA").save(buf, format=fmt, quality=90)
    return buf.getvalue(), f"image/{fmt.lower()}"

# EXTEND: Smarter preprocessing for hard images:
#   - Auto-crop the bounding region of items (drop background).
#   - Draw a faint grid overlay before sending so the model self-locates.
#   - Send the image at two resolutions (full + zoomed centre) as two parts.


def _parse_final(text: str) -> int | None:
    m = re.search(r"FINAL ANSWER:\s*-?\d[\d,]*", text or "", re.IGNORECASE)
    if m:
        return int(re.sub(r"\D", "", m.group().split(":", 1)[1]))
    nums = re.findall(r"-?\d[\d,]*", (text or "").replace(",", ""))
    return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


async def answer(image_bytes: bytes, mime_type: str, question: str) -> int:
    img, m = _shrink(image_bytes, mime_type)

    # Fresh session per call so state never leaks between rows.
    session = await runner.session_service.create_session(
        app_name="ggg_kit", user_id="participant",
    )
    content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=question),
            types.Part.from_bytes(data=img, mime_type=m),
        ],
    )

    # `run_async` yields events (tool calls, intermediate text, ...). The final
    # assistant message is the last event with a text part.
    final_text = ""
    async for event in runner.run_async(
        user_id="participant", session_id=session.id, new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

    n = _parse_final(final_text)
    if n is None:
        raise ValueError(f"could not parse integer from: {final_text!r}")
    return n

# EXTEND: Reflection / self-critique loop.
#   1. Run the agent once, get answer A.
#   2. Send a follow-up: "You answered A. Look again. Is A too high, too low,
#      or right? Reply with FINAL ANSWER: <integer>."
#   3. Run again, parse, return.
# Two passes ~= 2× latency, often a big accuracy gain on dense images.

"""Discover every agent in `agents/`, run them all on the dataset, compare results.

Adding a new agent is a one-step thing: drop `agents/<your_name>.py` with:

    NAME = "your-name"
    DESCRIPTION = "one line"

    async def answer(image_bytes: bytes, mime_type: str, question: str) -> int:
        ...

Then run this file. No registration, no editing of run_agents.py.

Toggle the PARALLEL flag below to fan every (agent, question) call out at once
(faster, messier logs). Default is sequential — easier to watch.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import pkgutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

# Silence google-genai's "non-text parts in response" noise. It fires every
# time the model returns a function_call alongside text (i.e. nearly every
# ADK tool-using turn) and is purely cosmetic.
warnings.filterwarnings("ignore", message=".*non-text parts.*")
for _name in ("google_genai", "google_genai.types", "google.genai", "google.genai.types"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import agents
from trace_logger import start_trace, stop_trace

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------

PARALLEL = False
# False → run one (agent, question) at a time. Slowest, easiest to follow.
# True  → fan every (agent, question) out concurrently. Fastest, logs interleave.

COMPETITION = False
# False → run on the practice dataset (test_images/ + dataset.json, with known answers).
# True  → run on the competition dataset (competition_dataset/ + competition_dataset.json).
#          Set this to True once you receive the competition files at the end of the hackathon.

# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
PRACTICE_DATASET_PATH = ROOT / "test_images" / "dataset.json"
COMPETITION_DATASET_PATH = ROOT / "competition_dataset" / "competition_dataset.json"
SUBMISSIONS_DIR = ROOT / "submissions"
LOGS_DIR = ROOT / "logs"

AnswerFn = Callable[[bytes, str, str], Awaitable[int]]


@dataclass
class AgentSpec:
    name: str
    description: str
    answer_fn: AnswerFn
    results: list[dict[str, Any]] = field(default_factory=list)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def discover_agents() -> list[AgentSpec]:
    """Import every module in the `agents/` package and pick up the ones that
    expose an `answer` coroutine."""
    specs: list[AgentSpec] = []
    for info in pkgutil.iter_modules(agents.__path__):
        mod = importlib.import_module(f"agents.{info.name}")
        if not asyncio.iscoroutinefunction(getattr(mod, "answer", None)):
            print(f"  · skipping agents/{info.name}.py — no `async def answer(...)`")
            continue
        specs.append(
            AgentSpec(
                name=getattr(mod, "NAME", info.name),
                description=getattr(mod, "DESCRIPTION", ""),
                answer_fn=mod.answer,
            )
        )
    return specs


async def run_one(spec: AgentSpec, row: dict, image_bytes: bytes, mime: str) -> None:
    label = f"[{spec.name:<14}] round {row['id']:>2}"
    print(f"{_ts()}  {label}  ▶ starting…", flush=True)
    log_path = start_trace(LOGS_DIR, spec.name, row["id"])
    t0 = time.monotonic()
    try:
        n = await spec.answer_fn(image_bytes, mime, row["question"])
        elapsed = time.monotonic() - t0
        truth = row.get("exact_count")
        if truth is not None:
            acc = max(0.0, 1.0 - abs(n - truth) / truth)
            print(
                f"{_ts()}  {label}  ✓ answer={n:<5} truth={truth:<5} "
                f"acc={acc:.2f}  ({elapsed:.1f}s)",
                flush=True,
            )
            spec.results.append(
                {
                    "id": row["id"],
                    "answer": n,
                    "truth": truth,
                    "accuracy": round(acc, 3),
                    "elapsed_s": round(elapsed, 2),
                }
            )
        else:
            print(
                f"{_ts()}  {label}  ✓ answer={n:<5}  ({elapsed:.1f}s)",
                flush=True,
            )
            spec.results.append(
                {
                    "id": row["id"],
                    "answer": n,
                    "elapsed_s": round(elapsed, 2),
                }
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        msg = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"{_ts()}  {label}  ✗ ERROR ({elapsed:.1f}s): {msg}", flush=True)
        spec.results.append({"id": row["id"], "error": str(e), "elapsed_s": round(elapsed, 2)})
    finally:
        stop_trace(spec.name, row["id"])
        print(f"{_ts()}  {label}  📄 log → {log_path.relative_to(ROOT)}", flush=True)


async def run_sequential(specs: list[AgentSpec], rows: list[dict], image_dir: Path = ROOT) -> None:
    for row in rows:
        image_path = image_dir / row["image"]
        mime = "image/png" if image_path.suffix == ".png" else "image/jpeg"
        image_bytes = image_path.read_bytes()
        truth_str = f"  (truth={row['exact_count']})" if "exact_count" in row else ""
        print(f"\n── round {row['id']}: {row['question']}{truth_str}")
        for spec in specs:
            await run_one(spec, row, image_bytes, mime)


async def run_parallel(specs: list[AgentSpec], rows: list[dict], image_dir: Path = ROOT) -> None:
    print(f"\nfanning out {len(specs)} agent(s) × {len(rows)} round(s) "
          f"= {len(specs) * len(rows)} concurrent calls…\n")
    tasks = []
    for row in rows:
        image_path = image_dir / row["image"]
        mime = "image/png" if image_path.suffix == ".png" else "image/jpeg"
        image_bytes = image_path.read_bytes()
        for spec in specs:
            tasks.append(run_one(spec, row, image_bytes, mime))
    await asyncio.gather(*tasks)


def print_summary(specs: list[AgentSpec], total_rows: int, total_elapsed: float) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY" + ("  (competition mode — answers only, no scoring)" if COMPETITION else ""))
    print("=" * 78)
    name_w = max(8, max(len(s.name) for s in specs))
    if COMPETITION:
        print(f"  {'agent':<{name_w}}   answers   errors   avg_call_s")
        print(f"  {'-' * name_w}   -------   ------   ----------")
        for s in specs:
            answered = sum(1 for r in s.results if "answer" in r)
            errors = sum(1 for r in s.results if "error" in r)
            elapseds = [r["elapsed_s"] for r in s.results if "elapsed_s" in r]
            avg_el = sum(elapseds) / len(elapseds) if elapseds else 0.0
            print(
                f"  {s.name:<{name_w}}   {answered:>4}/{total_rows:<2}    {errors:>4}     {avg_el:>7.1f}s"
            )
    else:
        print(f"  {'agent':<{name_w}}   avg_acc   exact   errors   avg_call_s")
        print(f"  {'-' * name_w}   -------   -----   ------   ----------")
        for s in specs:
            accs = [r["accuracy"] for r in s.results if "accuracy" in r]
            exact = sum(1 for r in s.results if r.get("accuracy") == 1.0)
            errors = sum(1 for r in s.results if "error" in r)
            elapseds = [r["elapsed_s"] for r in s.results if "elapsed_s" in r]
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            avg_el = sum(elapseds) / len(elapseds) if elapseds else 0.0
            print(
                f"  {s.name:<{name_w}}   {avg_acc:>7.3f}   "
                f"{exact:>2}/{total_rows:<2}    {errors:>4}     {avg_el:>7.1f}s"
            )
    print(f"\n  total wall time: {total_elapsed:.1f}s "
          f"({'parallel' if PARALLEL else 'sequential'})")


def write_submissions(specs: list[AgentSpec]) -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    print()
    for s in specs:
        prefix = "COMPETITION" if COMPETITION else "TEST"
        out = SUBMISSIONS_DIR / f"{prefix}_submission_{s.name}.json"
        out.write_text(
            json.dumps(
                {"agent": s.name, "description": s.description, "results": s.results},
                indent=2,
            )
        )
        print(f"  wrote {out.relative_to(ROOT)}")


async def main() -> None:
    specs = discover_agents()
    if not specs:
        print("no agents found — drop a file in agents/ following the contract in agents/__init__.py")
        return

    print(f"discovered {len(specs)} agent(s):")
    for s in specs:
        print(f"  · {s.name}: {s.description}")

    if COMPETITION:
        if not COMPETITION_DATASET_PATH.exists():
            print(f"\n  ✗ competition dataset not found at {COMPETITION_DATASET_PATH.relative_to(ROOT)}")
            print("    Copy the competition_dataset.json and images into competition_dataset/ and retry.")
            return
        dataset_path = COMPETITION_DATASET_PATH
        dataset = json.loads(dataset_path.read_text())
        print("\n  ★ COMPETITION MODE — running on the competition dataset (no answers, no scoring)")
    else:
        dataset_path = PRACTICE_DATASET_PATH
        dataset = json.loads(dataset_path.read_text())

    image_dir = dataset_path.parent
    rows = dataset["examples"]

    overall_t0 = time.monotonic()
    if PARALLEL:
        await run_parallel(specs, rows, image_dir)
    else:
        await run_sequential(specs, rows, image_dir)
    overall_elapsed = time.monotonic() - overall_t0

    print_summary(specs, len(rows), overall_elapsed)
    write_submissions(specs)


if __name__ == "__main__":
    asyncio.run(main())

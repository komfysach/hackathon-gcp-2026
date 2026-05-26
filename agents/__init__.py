"""Agent implementations.

Drop a new `<name>.py` file in this folder that exposes:

    NAME        = "your-agent-name"
    DESCRIPTION = "one-line summary"

    async def answer(image_bytes: bytes, mime_type: str, question: str) -> int:
        ...

`run_agents.py` (in the parent folder) auto-discovers everything in here that
follows that contract and runs it on the dataset.

See `baseline.py` for the simplest possible example, or `adk_example.py` for
an ADK + tools agent with extension hints.
"""

# Google Guessing Game — Participant Kit (macOS / Linux)

Build an agent that counts things in images. The agent that scores best on a hidden test set wins.

## Setup

You need:
- a GCP project
- [`gcloud` CLI](https://cloud.google.com/sdk/docs/install)

```bash
# 1. Log in to gcloud (opens a browser)
gcloud auth login

# 2. Pick a project (must be the lowercase PROJECT_ID, not the display NAME)
gcloud projects list                              # find your PROJECT_ID
# Or create a new one:
# gcloud projects create my-ggg-project
gcloud config set project YOUR_PROJECT_ID

# 3. Enable Vertex AI (wait ~1–2 min after this for it to fully propagate)
gcloud services enable aiplatform.googleapis.com

# 4. Auth the Python SDK with the same account (opens a browser again)
gcloud auth application-default login

# 5. Rename .env.example to `.env` and set GOOGLE_CLOUD_PROJECT to YOUR_PROJECT_ID

# 6. Install uv (one-time; auto-downloads Python 3.13)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 7. Create the venv + install deps
uv venv --python 3.13
source .venv/bin/activate                         # re-run in every new terminal
uv pip install -r requirements.txt
```

> Why two browser logins? `gcloud auth login` authenticates the **CLI** so you can run `gcloud ...` commands. `gcloud auth application-default login` writes credentials to disk that **Python SDKs** (and this kit) read automatically.

## Run

```bash
python run_agents.py
```

This auto-discovers every agent in `agents/`, runs each one on the 10 practice images in `test_images/`, streams progress to the terminal, and writes `submissions/TEST_submission_<agent>.json` per agent. At the end it prints a summary table comparing accuracy, exact matches, error count, and average call time.

By default agents run **sequentially** (one call at a time, easiest to follow). Flip `PARALLEL = True` near the top of `run_agents.py` to fan every (agent, question) pair out concurrently — much faster, but log lines will interleave.

## Build your agent

Open `agents/adk_example.py`. Read it once. Look for `# EXTEND:` comments — they mark every place where you can plug in something smarter (better prompts, more tools, multi-agent ensembles, self-critique loops, stronger model).

To add a new agent: copy `agents/adk_example.py` to `agents/my_agent.py` and edit. The file just needs to expose `NAME`, `DESCRIPTION`, and an `async def answer(image_bytes, mime_type, question) -> int`. `run_agents.py` will pick it up automatically the next time you run it.

## Competition mode

At the end of the hackathon you'll receive a `competition_dataset.json` and a set of images. To run on the competition dataset:

1. Copy the images and `competition_dataset.json` into the `competition_dataset/` folder.
2. Set `COMPETITION = True` near the top of `run_agents.py`.
3. Run `python run_agents.py` — it will use the competition dataset instead of the practice set. No answers or scoring will be shown (the JSON has no answers).

## Submitting

After running in competition mode, email the resulting `submissions/COMPETITION_submission_*.json` to **isaac@nextnovate.com** with your team name + a one-paragraph description of your strategy.

## Troubleshooting

- **`PermissionDenied: Vertex AI API has not been used`** — run `gcloud services enable aiplatform.googleapis.com`. If it appears mid-run after a successful enable, the API just hasn't fully propagated; wait 1–2 minutes and re-run.
- **`Permission denied on resource project your-gcp-project-id`** — you edited `.env.example` instead of `.env`, or never replaced the placeholder. Run `cat .env` and check.
- **`INVALID_ARGUMENT` from `gcloud config set project`** — you used the display NAME instead of the PROJECT_ID. Run `gcloud projects list` and use the value in the **PROJECT_ID** column (lowercase, hyphens only).
- **`could not parse integer from agent output`** — the model didn't end with `FINAL ANSWER: <integer>`. Check the `reasoning` field in the submission file and tighten your instruction.

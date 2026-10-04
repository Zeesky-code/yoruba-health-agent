#!/usr/bin/env bash
# Upload the app and the files it needs to the Hugging Face Space.
# Needs `hf auth login` (write token) and a COHERE_API_KEY secret set on the Space.
set -euo pipefail

SPACE="${SPACE:-Zeeskylaw/yoruba-health-agent}"
cd "$(dirname "$0")/.."

hf repos create "$SPACE" --repo-type space --space-sdk gradio --exist-ok
hf upload "$SPACE" . . --repo-type space \
  --include "app.py" --include "requirements.txt" --include "agent/*.py" \
  --include "corpus/corpus.jsonl" --include "evals/questions.jsonl" \
  --commit-message "deploy: $(git rev-parse --short HEAD)"
hf upload "$SPACE" space/README.md README.md --repo-type space \
  --commit-message "deploy: Space README"
echo "https://huggingface.co/spaces/$SPACE"

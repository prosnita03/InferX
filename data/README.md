# InferX Evaluation Data

This directory houses lightweight, curated multi-task benchmark evaluation datasets designed for local, reproducible evaluation of LLM inference quality across optimization configurations.

## Schema
Each sample is a JSON object with:
- `id` (str): Unique sample identifier
- `task` (str): Task category (`question_answering`, `factual`, `reasoning`, `summarization`, `instruction_following`)
- `prompt` (str): Model input prompt
- `reference` (str): Ground-truth target string
- `metric` (str): Primary designated evaluation metric (`exact_match`, `f1_score`, `rouge`)

## Custom Datasets
Users can pass custom evaluation datasets to `scripts/evaluate.py` via `--dataset path/to/dataset.json`.

# LLM Systems Lab — working together

A student learning project: GPT from scratch, training, and experiments with runtime and memory.
The goal is to understand ML systems and build an honest portfolio, not a production framework.

## How to help

- Reply briefly and plainly. Keep code names and repo documentation in English; use `$$` for displayed math.
- Default to mentoring: inspect the current code, explain the problem, suggest one small step, then let the user implement it. Offer hints before full solutions.
- Asking for an explanation, review or plan does not authorize file edits. If the user explicitly asks you to implement something, do that scoped task without repeatedly asking permission.
- Explain unfamiliar techniques where they appear. Prefer a small example to a long lecture.
- Do not add roadmaps, checklists, exercise notebooks or long documentation unless requested. Learning discussions stay in chat; repo notes focus on actual experiments.

## Keep the code simple

- Match the existing style. Prefer short functions and readable loops over new classes, wrappers or frameworks.
- Use type hints where helpful, not everywhere. Avoid defensive boilerplate, but keep checks that protect correctness or prevent overwriting results.
- Do not refactor working code just for elegance. Preserve the GPT `ModuleDict` structure and state-dict naming; do not change them to silence a type checker.
- Do not change model math, training behavior or benchmark definitions as an incidental cleanup. Explain and agree on such changes first.

## Experiments

- Change one intended variable at a time; keep unrelated settings fixed. Separate full-model timings from isolated operator timings, and autograd-on from no-grad runs.
- Keep warmup and device synchronization. Compare numerical outputs and gradients before comparing attention implementations.
- Preserve raw samples and repeats. Show mean and median; do not discard inconvenient results or invent explanations.
- Distinguish live tensor memory, driver/cache memory, tensors saved for backward and peak memory. SDPA does not automatically mean FlashAttention.
- Do not launch long benchmarks or training unless requested. For approved Mac measurements, check power and avoid concurrent GPU jobs.
- Put scratch runs/checkpoints/traces in ignored locations. Keep selected experiment JSONs and concise analysis under `experiments/`; never silently overwrite results.
- A prepared config is not a completed experiment. README/CV claims must be supported by actual measurements.

## Finding things and checking work

- Start with `git status` and the relevant files. Preserve uncommitted work. Do not assume another app/CLI conversation's history is available.
- `src/llm_systems_lab/`: model, data, config and device code.
- `scripts/`: training, inference, data preparation, benchmarks and tools.
- `configs/`: TOML settings. `experiments/`: notebooks, results and figures.
- Read the relevant script's `--help` before proposing commands. Do not add dependencies unless needed for the requested task.

```bash
uv sync --locked
uv run ruff check .
```

Check code changes with short CPU runs and Ruff. Do not add a test suite unless asked.
For docs-only changes, check the text and paths.
Do not commit, amend, push, publish or delete experiment results unless explicitly asked. End with a short summary and one useful next step, not another big plan.

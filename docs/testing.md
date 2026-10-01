# Testing and evaluation

TripMate AI is checked in two ways:

1. **Automated tests** check that the code behaves correctly. They need no API keys.
2. **Evaluation** checks how well the system makes decisions (guardrail, routing, trip details, answers).

## Automated tests

```bash
pip install -r requirements-dev.txt
pytest
```

**162 tests pass** without any API keys or services. They run the **real compiled LangGraph workflow**, with three replacements:

- a scripted (fake) LLM
- fake MCP tools that return exactly the same shape as the real MCP adapter
- an in-memory checkpointer instead of PostgreSQL

| Test file | What it checks |
|---|---|
| `test_guardrails.py` | All three guardrail layers, including injection rules, false positives and LLM failure (fail open) |
| `test_workflow.py` | Routing, agent selection, follow-up memory, human approval and resume, agent failures |
| `test_redis_cache.py` | Cache hit and miss, key normalisation, TTL, never caching failures, Redis down and the 30-second skip |
| `test_api.py` | Every endpoint and error code (`400`, `404`, `409`, `503`, `500`), and that no secrets leak |
| `test_observability.py` | Secret removal, trace events, dashboard values, LangSmith on and off |
| `test_evaluation.py` | The evaluation harness and its metrics |

### Tests with real services

`tests/test_real_services.py` (7 tests) runs against a real PostgreSQL and Redis. They're skipped unless you set:

```bash
export TRIPMATE_TEST_DATABASE_URL=postgresql://...   # use a throw-away database
export TRIPMATE_TEST_REDIS_URL=redis://...
pytest tests/test_real_services.py
```

## Evaluation

```bash
python -m evaluation.run                     # offline: fake LLM, no keys
python -m evaluation.run --mode live         # real Groq model, fake MCP tools (needs GROQ_API_KEY)
python -m evaluation.run --mode live --judge # also adds LLM-as-a-judge scores
```

Add `--real-mcp` to live mode to call the real MCP tools instead of the fake ones.

### Dataset

`evaluation/dataset.py` has **20 hand-written test cases**: valid trips, blocked requests (off-topic, injection, unsafe) and multi-turn follow-ups. Each case says whether it should be allowed, which agents must or must not run, and which trip details should be extracted.

The set is small on purpose. It's good for spotting regressions, but it isn't real user traffic and can't prove overall quality.

### Modes

| Mode | What it measures |
|---|---|
| **Offline** (default) | Input validation, rule-based guardrail, valid structured output and response format. Results that need a real model are reported as `skipped`. |
| **Live** | Everything above, plus the LLM guardrail, supervisor routing, extracted trip details (including follow-ups), simple answer checks and latency percentiles |
| **Live + judge** | An LLM scores each final answer for **relevance**, **groundedness** (based only on the agents' data?) and **completeness** |

The judge is the same model family as the writer, so it can be biased toward its own answers. Use its scores to compare runs, not as an absolute grade.

### Results

Reports are saved to `evaluation/results/` (ignored by git). The command exits with a non-zero code if any metric fails, so it can be used as a check in CI.

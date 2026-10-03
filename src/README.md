# Completed lab implementation

The seven scaffold modules are implemented. `responses.py` supplies the shared deterministic response policy and common prompt/token accounting.

- Baseline keeps only per-thread history.
- Advanced adds a durable `User.md` per user and a bounded summary per thread.
- Offline is the default and requires no API key or provider SDK.
- Live is optional, uses a LangChain chat model directly, and supports `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
- Profile extraction and summarization remain deterministic in both modes.

From the repository root:

```powershell
.\.venv\Scripts\python.exe src/benchmark.py --json-output benchmark_results.json
.\.venv\Scripts\python.exe -m pytest src/test_agents.py -v
```

For live mode, install `requirements-live.txt`, set provider credentials in `.env`, and pass `--live` to the benchmark. Setting `LLM_LIVE=true` enables live mode for directly constructed agents; `force_offline=True` always overrides it.

The benchmark isolates its temporary state. Direct use of AdvancedAgent writes durable profiles under the configured `state/profiles/` directory. See the root README, `STEP8.md` for the Guide step 8 answer, and `RESULTS.md` for detailed accounting definitions, limits, and measured results.

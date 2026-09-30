# Contributing

Thanks for taking a look. This is a personal project, so the process is light.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 or 3.12. Everything runs offline on synthetic data; no API keys or downloads are needed.

## Before you open a pull request

```bash
ruff format src tests && ruff check src tests   # lint and format
mypy src                                        # types
pytest                                          # tests, including the smoke training run
```

CI runs the same commands on Python 3.11 and 3.12.

## What makes a good change here

- **No leakage.** Any feature built from the target must use values at least `min_lag` days old. Add a case to `tests/unit/test_features.py::TestNoLeakage` if you add a feature family.
- **Numbers need a command.** If a change affects results, say which command produced the numbers you quote (for example `python -m src.train --smoke`) and on which data.
- **Say what it does not do.** Update the Limitations section of the README if your change has limits.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`), one logical change per commit.

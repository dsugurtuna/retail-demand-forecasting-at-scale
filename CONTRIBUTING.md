# Contributing to Retail Demand Forecasting at Scale

Thank you for your interest in contributing to this project! This document provides
guidelines and best practices for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

This project follows a Code of Conduct that all contributors are expected to follow.
Please be respectful, inclusive, and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional, for container-based development)
- Make (optional, for convenience commands)

### Finding Issues to Work On

1. Check the [Issues](../../issues) page for open issues
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on an issue before starting work to avoid duplication

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/retail-demand-forecasting-at-scale.git
cd retail-demand-forecasting-at-scale

# Add upstream remote
git remote add upstream https://github.com/dsugurtuna/retail-demand-forecasting-at-scale.git
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests to verify everything works
pytest tests/ -v

# Run linting
ruff check src tests
black --check src tests
```

## Making Changes

### Branch Naming Convention

Use descriptive branch names:

```
feature/add-new-model-type
bugfix/fix-prediction-cache
docs/update-api-reference
refactor/optimize-feature-pipeline
```

### Commit Message Format

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add batch prediction endpoint
fix(models): handle missing features gracefully
docs(readme): update installation instructions
test(metrics): add WRMSSE calculation tests
```

### Code Changes Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit:**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

3. **Keep your branch updated:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Test edge cases and error conditions
- Aim for >80% code coverage

Example test:
```python
import pytest
from src.evaluation.metrics import rmse

class TestRMSE:
    def test_rmse_perfect_predictions(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert rmse(y_true, y_pred) == pytest.approx(0.0)
    
    def test_rmse_with_errors(self):
        """RMSE should calculate correctly with errors."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)
```

## Pull Request Process

### Before Submitting

1. [ ] Tests pass locally (`pytest tests/ -v`)
2. [ ] Code is formatted (`black src tests`)
3. [ ] Linting passes (`ruff check src tests`)
4. [ ] Type hints are correct (`mypy src`)
5. [ ] Documentation is updated if needed
6. [ ] Commit messages follow conventions

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed my code
- [ ] Added comments for complex code
- [ ] Updated documentation
- [ ] No new warnings
```

### Review Process

1. Create PR against `main` branch
2. Wait for CI checks to pass
3. Request review from maintainers
4. Address review feedback
5. Squash commits if requested
6. Maintainer merges after approval

## Style Guide

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all public functions
- Use docstrings for all public modules, classes, and functions

```python
def predict(
    self,
    X: pd.DataFrame,
    return_interval: bool = False
) -> np.ndarray:
    """
    Generate predictions.
    
    Args:
        X: Feature DataFrame with shape (n_samples, n_features)
        return_interval: Whether to return prediction intervals
        
    Returns:
        Array of predictions with shape (n_samples,)
        
    Raises:
        ValueError: If X contains invalid features
        
    Example:
        >>> model = LightGBMForecaster()
        >>> predictions = model.predict(X_test)
    """
    ...
```

### Imports

Order imports as follows:
1. Standard library
2. Third-party packages
3. Local modules

```python
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.models.base import BaseForecaster
from src.utils.config import Config
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Type variables: `T`, `K`, `V`

### Documentation

- Use NumPy-style docstrings
- Include type hints in function signatures
- Document all public APIs
- Add examples for complex functions

## Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Open a new discussion for general questions
3. Open an issue for specific bugs or features

Thank you for contributing! 🎉

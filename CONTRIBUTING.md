# Contributing to DocSynthesis-V1

Thank you for your interest in contributing to DocSynthesis-V1! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow best practices

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/docsynthesis-v1/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Error messages and logs

### Suggesting Enhancements

1. Open an issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered
   - Implementation complexity estimate

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Update documentation
7. Commit with clear messages: `git commit -m "Add feature: description"`
8. Push to your fork: `git push origin feature/your-feature`
9. Create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/docsynthesis-v1.git
cd docsynthesis-v1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use:
- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking
- **Google-style** docstrings

Run before committing:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/unit/test_ocr.py
```

## Documentation

- Update docstrings for new functions/classes
- Update README.md if adding features
- Add examples for new functionality
- Update API documentation

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes

Example: `feat: add support for PDF batch processing`

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing! 🎉


# Contributing

Thanks for your interest in contributing to AI Developer Agent! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/your-username/ai-developer-agent.git
cd ai-developer-agent
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running Tests

```bash
# All unit tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific file
pytest tests/test_orchestrator.py -v

# Skip property-based tests (faster)
pytest -k "not Property"
```

## Code Style

- Python 3.11+
- Type hints on all public functions
- Docstrings on all public classes and functions
- Follow existing code patterns and naming conventions
- Use `async/await` for I/O operations
- Pydantic models for data validation

## Project Structure

- `src/agents/`  -  AI agent implementations (task reader, code finder, writer, reviewer)
- `src/clients/`  -  Direct REST API clients for GitLab and Bitbucket
- `src/config/`  -  Settings and MCP server configuration
- `src/pipeline/`  -  Pipeline orchestration, LLM routing, models
- `src/webhook/`  -  FastAPI webhook server and validation
- `src/utils/`  -  Git and Jira helper utilities
- `tests/`  -  Unit tests and property-based tests (Hypothesis)

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit with conventional commits: `feat: add new feature` or `fix: resolve bug`
6. Push and open a PR against `main`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add GitLab self-hosted support
fix: handle empty repository field gracefully
docs: update setup instructions
test: add property tests for token budget
refactor: extract LLM routing logic
```

## Adding a New Git Provider

1. Create a client in `src/clients/` (see `gitlab_client.py` as reference)
2. Add provider config to `src/config/settings.py`
3. Update `src/pipeline/orchestrator.py` to use the new client
4. Add tests in `tests/`
5. Update `.env.example` and `README.md`

## Adding a New LLM Provider

1. Add the provider to `src/pipeline/llm_router.py`
2. Update `src/config/settings.py` with any new settings
3. Add tests
4. Update `.env.example` and `README.md`

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Include relevant logs (redact any credentials)
- Specify your environment (OS, Python version, Docker version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

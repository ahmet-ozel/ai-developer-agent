# AI Developer Agent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-6E56CF)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous AI agent that reads Jira tasks, finds relevant code in your repository, generates changes with LLM, reviews them, and creates pull requests - all automatically.

## How It Works

```
Jira Task → TaskReader → CodeFinder → CodeWriter → CodeReviewer → Pull Request
```

1. **TaskReader** - Reads the Jira issue (summary, description, acceptance criteria, comments, linked issues) and estimates scope
2. **CodeFinder** - Locates relevant source files in the repository using the git provider API
3. **CodeWriter** - Generates code changes and tests using a strong LLM (GPT-4o, Claude, etc.)
4. **CodeReviewer** - Reviews the generated code for quality, security, and correctness
5. **PR Creation** - Creates a branch, commits changes, opens a pull request, and comments back on Jira

## Supported Platforms

| Platform | Method | Auth |
|---|---|---|
| **GitHub** | MCP Server (`@modelcontextprotocol/server-github`) | Personal Access Token |
| **GitLab** | Direct REST API (`httpx`) | Personal Access Token |
| **Bitbucket** | Direct REST API (`httpx`) | Atlassian API Token |

## Supported LLM Providers

| Provider | Models | Notes |
|---|---|---|
| **OpenAI** | gpt-4o, gpt-4o-mini, etc. | Default provider |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku, etc. | |
| **Google** | Gemini Pro, etc. | |
| **vLLM** | Any OpenAI-compatible model | Self-hosted, requires tool-calling support |

The agent uses a two-tier LLM system:
- **Fast tier** - For task reading, code finding (e.g., gpt-4o-mini)
- **Strong tier** - For code writing, code review (e.g., gpt-4o)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ahmet-ozel/jira-autonomous-coding-agent.git
cd jira-autonomous-coding-agent
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Jira Setup

1. Create a dedicated Jira user (e.g., `ai-developer-bot`)
2. Add a custom field named **"Repository"** (short text type) to your Jira project
3. Set the field value to the repository name (e.g., `my-backend-api`) on each issue
4. The agent discovers this field dynamically at runtime - no hardcoded field IDs

### 4. Run with Docker (Recommended)

```bash
docker-compose up -d
```

Or build and run manually:

```bash
docker build -t ai-developer-agent .
docker run --env-file .env -p 8000:8000 ai-developer-agent
```

### 5. Run Locally

```bash
uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Trigger Modes

### Polling Mode (Default)

The agent polls Jira every N seconds for tasks assigned to the bot user. No webhook or public URL needed.

```env
TRIGGER_MODE=polling
POLL_INTERVAL_SECONDS=30
JIRA_PROJECT_KEY=RP
```

### Webhook Mode

For real-time triggering via Jira webhooks:

```env
TRIGGER_MODE=webhook
JIRA_WEBHOOK_SECRET=your-random-secret
```

In Jira: **Project Settings → Webhooks → Create**
- URL: `http://your-server:8000/webhook/jira`
- Events: Issue Updated, Issue Created
- Secret: same as `JIRA_WEBHOOK_SECRET`

## Git Provider Setup

### GitHub

```env
GIT_PROVIDER=github
GITHUB_TOKEN=ghp_your_token
GITHUB_OWNER=your-username-or-org
```

Token scopes needed: `repo` (full repository access)

### GitLab

```env
GIT_PROVIDER=gitlab
GITLAB_TOKEN=glpat-your_token
GITLAB_GROUP=your-group
GITLAB_URL=https://gitlab.com  # or your self-hosted URL
```

Token scopes needed: `api`

### Bitbucket

```env
GIT_PROVIDER=bitbucket
BITBUCKET_WORKSPACE=your-workspace
BITBUCKET_USERNAME=your-email@example.com
BITBUCKET_APP_PASSWORD=your-atlassian-api-token
```

> **Note:** Bitbucket app passwords were deprecated in September 2025. Use Atlassian API tokens with Bitbucket scopes instead. The `BITBUCKET_USERNAME` should be your Atlassian account email.

## LLM Configuration

### OpenAI (Default)

```env
LLM_FAST_PROVIDER=openai
LLM_FAST_MODEL=gpt-4o-mini
LLM_FAST_API_KEY=sk-...

LLM_STRONG_PROVIDER=openai
LLM_STRONG_MODEL=gpt-4o
LLM_STRONG_API_KEY=sk-...
```

### Anthropic

```env
LLM_FAST_PROVIDER=anthropic
LLM_FAST_MODEL=claude-3-haiku-20240307
LLM_FAST_API_KEY=sk-ant-...

LLM_STRONG_PROVIDER=anthropic
LLM_STRONG_MODEL=claude-3-5-sonnet-20241022
LLM_STRONG_API_KEY=sk-ant-...
```

### vLLM (Self-Hosted)

```env
LLM_FAST_PROVIDER=vllm
LLM_FAST_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
LLM_FAST_API_KEY=not-needed
LLM_FAST_ENDPOINT=http://localhost:8080/v1
```

> The model must support tool-calling. Start vLLM with `--enable-auto-tool-choice --tool-call-parser hermes`.

### Fallback Chain

Configure backup providers in case the primary fails:

```env
LLM_FALLBACK_CHAIN=["anthropic", "google"]
```

## Confluence Integration (Optional)

The agent can automatically publish code review documentation to Confluence:

```env
CONFLUENCE_ENABLED=true
CONFLUENCE_URL=https://your-site.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=your-confluence-token
CONFLUENCE_SPACE_KEY=DEV
CONFLUENCE_PARENT_PAGE_ID=12345
```

## Configuration Reference

### Pipeline Settings

| Variable | Default | Description |
|---|---|---|
| `DRY_RUN` | `false` | Skip all Git/Jira writes, log only |
| `MAX_REVIEW_RETRIES` | `2` | Max CodeWriter → CodeReviewer iterations |
| `MAX_FILE_CHANGES` | `15` | Max files changed per task |
| `MAX_FILES_PER_TASK` | `10` | Max files read by CodeFinder |
| `MAX_CONTEXT_TOKENS` | `100000` | Token budget for LLM context |
| `MAX_FILE_SIZE_KB` | `100` | Skip files larger than this |
| `BRANCH_PATTERN` | `feature/{issue_key}-ai` | Branch name template |
| `AUTO_CREATE_PR` | `true` | Create PR automatically |
| `PR_AUTO_ASSIGN_REVIEWER` | `false` | Assign task reporter as PR reviewer |
| `PR_DRAFT_MODE` | `true` | Create PRs as drafts |

### Task Filtering

| Variable | Default | Description |
|---|---|---|
| `SKIP_TASK_TYPES` | `[]` | Issue types to skip (e.g., `["Epic"]`) |
| `ALLOWED_TASK_TYPES` | `[]` | Whitelist of issue types (empty = all) |

### LLM Tier Overrides

| Variable | Default | Description |
|---|---|---|
| `TASK_READER_LLM_TIER` | `fast` | LLM tier for reading tasks |
| `CODE_FINDER_LLM_TIER` | `fast` | LLM tier for finding code |
| `CODE_WRITER_LLM_TIER` | `strong` | LLM tier for writing code |
| `CODE_REVIEWER_LLM_TIER` | `strong` | LLM tier for reviewing code |

## Architecture

```
ai-developer-agent/
├── src/
│   ├── agents/              # AI agent implementations
│   │   ├── task_reader.py   # Reads Jira issues, estimates scope
│   │   ├── code_finder.py   # Finds relevant files in repository
│   │   ├── code_writer.py   # Generates code changes
│   │   └── code_reviewer.py # Reviews generated code
│   ├── clients/             # Direct REST API clients
│   │   ├── gitlab_client.py # GitLab API (no MCP)
│   │   └── bitbucket_client.py # Bitbucket API (no MCP)
│   ├── config/
│   │   ├── settings.py      # Pydantic settings from .env
│   │   └── mcp_servers.py   # MCP server config builder
│   ├── pipeline/
│   │   ├── orchestrator.py  # Main pipeline orchestration
│   │   ├── llm_router.py    # Two-tier LLM routing
│   │   ├── models.py        # Pydantic data models
│   │   ├── confluence_publisher.py # Confluence docs
│   │   ├── token_budget.py  # Token budget management
│   │   ├── retry.py         # Retry with backoff
│   │   └── logging.py       # Structured logging
│   ├── webhook/
│   │   ├── server.py        # FastAPI webhook endpoint
│   │   ├── validators.py    # Webhook signature validation
│   │   ├── task_lock.py     # In-memory task deduplication
│   │   └── models.py        # Webhook event models
│   ├── utils/
│   │   ├── git_helpers.py   # Branch naming, PR helpers
│   │   └── jira_helpers.py  # Jira API helpers
│   ├── app.py               # MCPApp factory, pipeline entry
│   └── main.py              # FastAPI app factory
├── tests/                   # Unit + property-based tests
├── scripts/                 # Utility scripts
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

### Key Design Decisions

- **GitHub uses MCP server**, GitLab and Bitbucket use direct REST API clients (the `@modelcontextprotocol/server-gitlab` MCP server has critical zod@4 compatibility issues)
- **Two-tier LLM system** - fast models for reading/finding, strong models for writing/reviewing
- **Dynamic Jira field discovery** - the "Repository" custom field ID is discovered at runtime, not hardcoded
- **In-memory task lock** - prevents duplicate processing; requires single-worker deployment (`--workers 1`)
- **Token budget management** - prevents LLM context overflow by tracking token usage

## Utility Scripts

```bash
# Check all service credentials
python scripts/check_credentials.py

# Run pipeline for a specific issue (no webhook needed)
python scripts/run_pipeline.py RP-1
python scripts/run_pipeline.py RP-1 --dry-run

# Poll Jira manually
python scripts/poll_jira.py --interval 30

# Set up test data (GitHub repo + Jira issue)
python scripts/setup_test_data.py
```

## End-to-End Workflow

1. Create a Jira issue with a clear description and acceptance criteria
2. Set the **Repository** custom field to the target repo name (e.g., `my-backend-api`)
3. Assign the issue to `ai-developer-bot`
4. The agent picks it up (via polling or webhook)
5. Pipeline runs: read task → find code → write changes → review → create PR
6. Agent comments on the Jira issue with the PR link
7. Issue is transitioned to "In Review"

## Testing

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run e2e tests (requires real credentials in .env)
pytest tests/e2e/ -v -m e2e

# Run tests in Docker
docker-compose run --rm test
```

## Troubleshooting

**Agent not picking up tasks**
- Verify `JIRA_BOT_USERNAME` matches the Jira user assigned to the issue
- Check that the "Repository" custom field exists and has a value
- In polling mode, check `JIRA_PROJECT_KEY` is set correctly

**LLM errors / timeouts**
- Configure `LLM_FALLBACK_CHAIN` with backup providers
- Reduce `MAX_FILES_PER_TASK` to lower token usage
- Check API key validity and rate limits

**Git authentication errors**
- GitHub: token needs `repo` scope
- GitLab: token needs `api` scope
- Bitbucket: use Atlassian API token (not legacy app password), username is your email

**Branch collision**
- Handled automatically with timestamp suffix retry

**Webhook not triggering**
- Verify the webhook URL is reachable from Jira
- Check `JIRA_WEBHOOK_SECRET` matches
- Confirm the issue is assigned to the bot user

**Single worker requirement**
- `TaskLock` is in-memory - use `--workers 1` with uvicorn
- For multi-replica deployments, implement a Redis-based distributed lock

## Docker

### Development

```bash
docker-compose up -d
# Source code is live-mounted at ./src:/app/src:ro
```

### Production

Remove the volume mount from `docker-compose.yml` and rebuild:

```bash
docker-compose build
docker-compose up -d
```

### Health Check

```
GET http://localhost:8000/health
```

## License

MIT
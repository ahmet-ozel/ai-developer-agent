# End-to-End Testing Guide

This guide walks you through setting up real services (no mocks) to test the AI Developer Agent pipeline end-to-end.

## Required Accounts

| Service | URL | Free Plan |
|---------|-----|-----------|
| Jira Cloud | https://www.atlassian.com/software/jira/free | Up to 10 users |
| GitHub / GitLab / Bitbucket |  -  | Free tier sufficient |
| ngrok (webhook mode only) | https://ngrok.com | Free plan works |

## Step 1: Jira Cloud Setup

### 1.1 Create Account
1. Go to https://www.atlassian.com/software/jira/free
2. Click "Get it free"  sign up with email
3. Choose a site name (e.g., `myteam.atlassian.net`)
4. Select "Scrum" or "Kanban" template

### 1.2 Create Bot User
1. In Jira: Settings  User Management  Invite users
2. Invite a new user as `ai-developer-bot`
3. Note the username  this goes in `.env` as `JIRA_BOT_USERNAME`

### 1.3 Get API Token
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. "Create API token"  name it  copy
3. Paste in `.env` as `JIRA_API_TOKEN`

### 1.4 Add "Repository" Custom Field
1. Jira  Project Settings  Fields
2. "Create custom field"  Type: "Short text"  Name: `Repository`
3. Add it to your project

## Step 2: Git Provider Setup

### GitHub
1. https://github.com/settings/tokens  "Generate new token (classic)"
2. Scopes: `repo` (all sub-options)
3. Copy token  `.env` as `GITHUB_TOKEN`

### GitLab
1. https://gitlab.com/-/user_settings/personal_access_tokens
2. Scopes: `api`
3. Copy token  `.env` as `GITLAB_TOKEN`

### Bitbucket
1. https://id.atlassian.com/manage-profile/security/api-tokens
2. Create an Atlassian API token with Bitbucket scopes
3. Copy token  `.env` as `BITBUCKET_APP_PASSWORD`
4. `BITBUCKET_USERNAME` should be your Atlassian account email

## Step 3: LLM Provider Setup

### OpenAI (Recommended)
1. https://platform.openai.com/api-keys  "Create new secret key"
2. Set in `.env`:
```env
LLM_FAST_PROVIDER=openai
LLM_FAST_MODEL=gpt-4o-mini
LLM_FAST_API_KEY=sk-...
LLM_STRONG_PROVIDER=openai
LLM_STRONG_MODEL=gpt-4o
LLM_STRONG_API_KEY=sk-...
```

## Step 4: Configure .env

```bash
cd ai-developer-agent
cp .env.example .env
# Edit .env  -  fill in all credentials
```

## Step 5: Verify Credentials

```bash
python scripts/check_credentials.py
```

All checks should pass. Fix any failures before proceeding.

## Step 6: Create Test Data

```bash
python scripts/setup_test_data.py
```

This creates a test repository with sample Python files and a Jira issue assigned to the bot user.

## Step 7: Run Tests

### Unit tests (mock-based, always works)
```bash
pytest tests/ --ignore=tests/e2e --ignore=tests/integration -q
```

### E2E tests (real APIs, credentials required)
```bash
pytest tests/e2e/ -v -m e2e
```

### Full pipeline test
```bash
# Start the server
uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000

# In polling mode: assign a Jira issue to the bot user
# The agent will pick it up automatically

# In webhook mode: also start ngrok
ngrok http 8000
# Set the ngrok URL as the Jira webhook URL
```

### DRY_RUN Mode
Start with `DRY_RUN=true`  -  logs everything but skips actual Git/Jira writes:
```env
DRY_RUN=true
```

Once verified, set `DRY_RUN=false` for real execution.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Jira 401 | Check API token, verify email + token are correct |
| GitHub 401 | Token may be expired, generate a new one |
| GitHub 403 | Token scopes insufficient, add `repo` scope |
| GitLab 401 | Token needs `api` scope |
| Bitbucket 401 | Use email as username, Atlassian API token as password |
| LLM timeout | Check model name and API key validity |
| Webhook not triggering | Verify ngrok URL, check Jira webhook is active |
| Pipeline error | Use `DRY_RUN=true` and check logs |

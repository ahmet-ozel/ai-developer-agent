#!/usr/bin/env python3
"""Create test data on real services for e2e testing.

Usage:
    python scripts/setup_test_data.py

Creates:
- GitHub: test repo with sample Python files (if GIT_PROVIDER=github)
- Jira: test project + issue with "repository" custom field

Requires .env to be configured with valid credentials.
Run scripts/check_credentials.py first to verify connectivity.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

TEST_REPO_NAME = "ai-agent-test-repo"
TEST_PROJECT_KEY = "AITEST"


# ── Sample files to put in the test repo ──────────────────────────────

SAMPLE_FILES = {
    "calculator.py": '''"""Simple calculator module for testing AI agent code changes."""


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of a and b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return the quotient of a and b.

    TODO: Handle division by zero properly.
    """
    return a / b
''',
    "user_service.py": '''"""User service module - intentionally has issues for the AI agent to fix."""

from dataclasses import dataclass


@dataclass
class User:
    name: str
    email: str
    age: int


def create_user(name: str, email: str, age: int) -> User:
    """Create a new user.

    TODO: Add input validation (email format, age range).
    """
    return User(name=name, email=email, age=age)


def get_user_display_name(user: User) -> str:
    """Return a display-friendly name."""
    return user.name


def is_adult(user: User) -> bool:
    """Check if user is an adult (18+)."""
    return user.age >= 18
''',
    "README.md": '''# AI Agent Test Repository

This repository is used for end-to-end testing of the AI Developer Agent.

## Files

- `calculator.py` - Simple calculator with a division-by-zero bug
- `user_service.py` - User service with missing input validation
''',
}


# ── GitHub ────────────────────────────────────────────────────────────


async def setup_github_repo() -> str | None:
    """Create test repo on GitHub with sample files. Returns repo URL."""
    token = os.getenv("GITHUB_TOKEN", "")
    owner = os.getenv("GITHUB_OWNER", "")

    if not token or not owner:
        print(f"  {RED}✗{RESET} GITHUB_TOKEN or GITHUB_OWNER not set")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        # Check if repo already exists
        resp = await client.get(f"https://api.github.com/repos/{owner}/{TEST_REPO_NAME}")
        if resp.status_code == 200:
            print(f"  {YELLOW}⚠{RESET} Repo {owner}/{TEST_REPO_NAME} already exists - skipping creation")
            return f"https://github.com/{owner}/{TEST_REPO_NAME}"

        # Create repo
        resp = await client.post(
            "https://api.github.com/user/repos",
            json={
                "name": TEST_REPO_NAME,
                "description": "Test repo for AI Developer Agent e2e testing",
                "private": True,
                "auto_init": True,
            },
        )
        if resp.status_code not in (201, 200):
            print(f"  {RED}✗{RESET} Failed to create repo: {resp.status_code} {resp.text[:200]}")
            return None

        print(f"  {GREEN}✓{RESET} Created repo: {owner}/{TEST_REPO_NAME}")

        # Wait a moment for GitHub to initialize
        await asyncio.sleep(2)

        # Add sample files
        for filename, content in SAMPLE_FILES.items():
            import base64
            resp = await client.put(
                f"https://api.github.com/repos/{owner}/{TEST_REPO_NAME}/contents/{filename}",
                json={
                    "message": f"Add {filename} for e2e testing",
                    "content": base64.b64encode(content.encode()).decode(),
                },
            )
            if resp.status_code in (200, 201):
                print(f"  {GREEN}✓{RESET} Added {filename}")
            else:
                print(f"  {RED}✗{RESET} Failed to add {filename}: {resp.status_code}")

        return f"https://github.com/{owner}/{TEST_REPO_NAME}"


# ── Jira ──────────────────────────────────────────────────────────────


async def setup_jira_issue() -> str | None:
    """Create a test issue on Jira Cloud. Returns issue key."""
    url = os.getenv("JIRA_URL", "").rstrip("/")
    username = os.getenv("JIRA_USERNAME", "")
    token = os.getenv("JIRA_API_TOKEN", "")
    bot_username = os.getenv("JIRA_BOT_USERNAME", "")

    if not all([url, username, token]):
        print(f"  {RED}✗{RESET} Jira credentials not set")
        return None

    auth = (username, token)

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Check if project exists
        resp = await client.get(
            f"{url}/rest/api/3/project/{TEST_PROJECT_KEY}",
            auth=auth,
        )
        if resp.status_code == 200:
            print(f"  {YELLOW}⚠{RESET} Project {TEST_PROJECT_KEY} already exists")
        else:
            print(f"  {YELLOW}⚠{RESET} Project {TEST_PROJECT_KEY} not found - you need to create it manually in Jira")
            print(f"      Go to: {url}/secure/admin/CreateProject!default.jspa")
            print(f"      Use key: {TEST_PROJECT_KEY}")

        # 2. Find the "repository" custom field
        resp = await client.get(
            f"{url}/rest/api/3/field",
            auth=auth,
        )
        repo_field_id = None
        if resp.status_code == 200:
            fields = resp.json()
            for f in fields:
                if f.get("name", "").lower() == "repository":
                    repo_field_id = f["id"]
                    print(f"  {GREEN}✓{RESET} Found 'repository' custom field: {repo_field_id}")
                    break
            if not repo_field_id:
                print(f"  {YELLOW}⚠{RESET} 'repository' custom field not found - create it in Jira project settings")
                print(f"      Type: Short text, Name: repository")

        # 3. Create test issue
        issue_data: dict = {
            "fields": {
                "project": {"key": TEST_PROJECT_KEY},
                "summary": "Add input validation to divide function",
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "The divide() function in calculator.py does not handle "
                                        "division by zero. Add proper validation that raises a "
                                        "ValueError with a descriptive message when b is zero. "
                                        "Also add unit tests for the fix."
                                    ),
                                }
                            ],
                        }
                    ],
                },
                "issuetype": {"name": "Task"},
            }
        }

        # Add repository field if found
        if repo_field_id:
            issue_data["fields"][repo_field_id] = TEST_REPO_NAME

        resp = await client.post(
            f"{url}/rest/api/3/issue",
            auth=auth,
            json=issue_data,
        )
        if resp.status_code in (200, 201):
            issue_key = resp.json()["key"]
            print(f"  {GREEN}✓{RESET} Created issue: {issue_key}")

            # 4. Assign to bot user if configured
            if bot_username:
                # Find bot account ID
                resp2 = await client.get(
                    f"{url}/rest/api/3/user/search?query={bot_username}",
                    auth=auth,
                )
                if resp2.status_code == 200 and resp2.json():
                    account_id = resp2.json()[0]["accountId"]
                    await client.put(
                        f"{url}/rest/api/3/issue/{issue_key}/assignee",
                        auth=auth,
                        json={"accountId": account_id},
                    )
                    print(f"  {GREEN}✓{RESET} Assigned {issue_key} to {bot_username}")
                else:
                    print(f"  {YELLOW}⚠{RESET} Bot user '{bot_username}' not found - assign manually")

            return issue_key
        else:
            print(f"  {RED}✗{RESET} Failed to create issue: {resp.status_code} {resp.text[:300]}")
            return None


# ── Main ──────────────────────────────────────────────────────────────


async def main() -> None:
    print(f"{BOLD}{'='*60}")
    print("  AI Developer Agent - Test Data Setup")
    print(f"{'='*60}{RESET}")

    provider = os.getenv("GIT_PROVIDER", "github")

    # Git repo
    print(f"\n{BOLD}Git Repository ({provider}){RESET}")
    repo_url = None
    if provider == "github":
        repo_url = await setup_github_repo()
    else:
        print(f"  {YELLOW}⚠{RESET} Auto-setup only supports GitHub for now. Create repo manually for {provider}.")

    # Jira issue
    print(f"\n{BOLD}Jira Issue{RESET}")
    issue_key = await setup_jira_issue()

    # Summary
    print(f"\n{BOLD}{'='*60}")
    print("  Setup Summary")
    print(f"{'='*60}{RESET}")
    if repo_url:
        print(f"  Repo: {repo_url}")
    if issue_key:
        print(f"  Issue: {issue_key}")
    print()
    print("  Next steps:")
    print("  1. Verify the issue in Jira has the 'repository' field set")
    print(f"  2. Assign the issue to your bot user ({os.getenv('JIRA_BOT_USERNAME', 'ai-developer-bot')})")
    print("  3. Set up Jira webhook pointing to your server")
    print("  4. Run: python scripts/check_credentials.py")
    print("  5. Run: uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    asyncio.run(main())

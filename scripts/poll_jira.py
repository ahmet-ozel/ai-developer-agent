#!/usr/bin/env python3
"""Jira polling mode - checks for new bot-assigned tasks periodically.

Usage:
    python scripts/poll_jira.py [--interval 30]

No webhook or ngrok needed. Polls Jira REST API every N seconds,
finds issues assigned to the bot user, and triggers the pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Track already-processed issues to avoid re-triggering
_processed: set[str] = set()


async def find_bot_tasks() -> list[dict]:
    """Query Jira for issues assigned to the bot user."""
    jira_url = os.getenv("JIRA_URL", "").rstrip("/")
    username = os.getenv("JIRA_USERNAME", "")
    token = os.getenv("JIRA_API_TOKEN", "")
    bot = os.getenv("JIRA_BOT_USERNAME", "ai-developer-bot")
    project_key = os.getenv("JIRA_PROJECT_KEY", "")
    project_filter = f'project = {project_key} AND ' if project_key else ''
    jql = f'{project_filter}assignee = "{bot}" AND status != Done ORDER BY updated DESC'

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{jira_url}/rest/api/3/search/jql",
            auth=(username, token),
            json={"jql": jql, "maxResults": 10, "fields": ["summary", "status", "assignee"]},
        )

    if resp.status_code != 200:
        logger.error("Jira search failed: %s %s", resp.status_code, resp.text[:200])
        return []

    data = resp.json()
    issues = data.get("issues", [])
    return issues


async def trigger_pipeline(issue_key: str) -> None:
    """Trigger the pipeline for a given issue key."""
    logger.info("Triggering pipeline for %s", issue_key)

    try:
        from src.config.settings import Settings
        from src.app import run_pipeline

        settings = Settings()  # type: ignore[call-arg]
        await run_pipeline(issue_key, settings)
        logger.info("Pipeline completed for %s", issue_key)
    except Exception:
        logger.exception("Pipeline failed for %s", issue_key)


async def poll_loop(interval: int) -> None:
    """Main polling loop."""
    logger.info(
        "Polling Jira every %ds for tasks assigned to '%s'",
        interval,
        os.getenv("JIRA_BOT_USERNAME", "ai-developer-bot"),
    )
    logger.info("DRY_RUN=%s", os.getenv("DRY_RUN", "false"))

    while True:
        try:
            issues = await find_bot_tasks()
            for issue in issues:
                key = issue["key"]
                summary = issue["fields"]["summary"]
                status = issue["fields"]["status"]["name"]

                if key in _processed:
                    continue

                logger.info("Found new task: %s - %s [%s]", key, summary, status)
                _processed.add(key)
                await trigger_pipeline(key)

        except Exception:
            logger.exception("Error during poll cycle")

        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll Jira for bot-assigned tasks")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()
    asyncio.run(poll_loop(args.interval))


if __name__ == "__main__":
    main()

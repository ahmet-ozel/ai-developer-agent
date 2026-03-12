#!/usr/bin/env python3
"""Run the pipeline for a specific Jira issue — no webhook needed.

Usage:
    python scripts/run_pipeline.py RP-1
    python scripts/run_pipeline.py RP-1 --dry-run

Directly triggers the full pipeline for the given issue key.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


async def main(issue_key: str, dry_run: bool) -> None:
    if dry_run:
        os.environ["DRY_RUN"] = "true"

    from src.config.settings import Settings
    from src.app import run_pipeline

    settings = Settings()  # type: ignore[call-arg]
    logger.info("Starting pipeline for %s (DRY_RUN=%s)", issue_key, settings.dry_run)

    try:
        await run_pipeline(issue_key, settings)
        logger.info("Pipeline completed for %s", issue_key)
    except Exception:
        logger.exception("Pipeline failed for %s", issue_key)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline for a Jira issue")
    parser.add_argument("issue_key", help="Jira issue key (e.g. RP-1)")
    parser.add_argument("--dry-run", action="store_true", help="Skip Git/Jira writes")
    args = parser.parse_args()
    asyncio.run(main(args.issue_key, args.dry_run))

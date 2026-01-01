import json
import time
from typing import Dict, Any


def log_interaction(logfile: str, interaction: Dict[str, Any]) -> None:
    """Append an interaction JSON object to `logfile` with a timestamp."""
    entry = {"ts": time.time(), "interaction": interaction}
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def collect_metrics(interaction: Dict[str, Any]) -> Dict[str, Any]:
    """Collect simple metrics from an interaction dict.

    Example metrics: answer length, number of sources, retrieval time (if provided).
    """
    metrics = {}
    ans = interaction.get("answer") or ""
    metrics["answer_chars"] = len(ans)
    metrics["num_sources"] = len(interaction.get("sources", []))
    if "retrieval" in interaction and isinstance(interaction["retrieval"], dict):
        # try to extract number of retrieved items
        results = interaction["retrieval"].get("results") or []
        metrics["retrieved_items"] = len(results)
    return metrics


__all__ = ["log_interaction", "collect_metrics"]

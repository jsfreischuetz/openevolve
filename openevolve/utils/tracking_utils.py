"""
Lightweight logging helpers for tracking token usage and metric convergence.

These helpers append CSV rows so the logs can be loaded directly with
`pandas.read_csv` without additional parsing.
"""

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

DEFAULT_TOKEN_LOG = "token_usage.csv"
DEFAULT_CONVERGENCE_LOG = "convergence_log.csv"


def resolve_log_dir(preferred: Optional[str] = None) -> Path:
    """Resolve the directory for structured logs and ensure it exists."""
    env_dir = os.environ.get("OPENEVOLVE_LOG_DIR")
    base = Path(preferred or env_dir or (Path.cwd() / "logs"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_row(file_path: Path, row: Dict[str, Any], field_order: Sequence[str]) -> None:
    """Append a single CSV row with a stable header."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    exists = file_path.exists()

    with file_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(field_order))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in field_order})


def log_token_usage(
    model_name: str,
    provider: str,
    usage: Optional[Dict[str, Any]],
    log_context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
) -> None:
    """
    Append a token-usage entry to the token log.

    Args:
        model_name: Name of the model used (e.g., "gpt-4o").
        provider: Provider string (e.g., "openai", "anthropic").
        usage: Raw usage dict returned by the client.
        log_context: Optional label describing the call site ("evolution", "llm_evaluation", etc.).
        metadata: Optional dictionary with keys like "iteration" or "program_id".
        log_dir: Optional override for the log directory.
    """
    if not usage:
        return

    base = resolve_log_dir(log_dir)
    field_order = [
        "timestamp",
        "model",
        "provider",
        "context",
        "iteration",
        "program_id",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "accepted_prediction_tokens",
        "reasoning_tokens",
        "rejected_prediction_tokens",
        "cached_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ]

    row: Dict[str, Any] = {
        "timestamp": time.time(),
        "model": model_name,
        "provider": provider,
        "context": log_context,
        "iteration": None,
        "program_id": None,
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "accepted_prediction_tokens": usage.get("accepted_prediction_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "rejected_prediction_tokens": usage.get("rejected_prediction_tokens"),
        "cached_tokens": usage.get("cached_tokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if isinstance(usage, dict)
        else None,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
    }

    if metadata:
        row["iteration"] = metadata.get("iteration", row["iteration"])
        row["program_id"] = metadata.get("program_id", row["program_id"])

    _write_row(base / DEFAULT_TOKEN_LOG, row, field_order)


def log_convergence(
    iteration: int,
    metric_name: str,
    metric_value: float,
    best_metric: float,
    delta_from_prev: Optional[float],
    delta_from_best: Optional[float],
    log_dir: Optional[str] = None,
) -> None:
    """Append a convergence-tracking entry to the convergence log."""
    base = resolve_log_dir(log_dir)
    field_order = [
        "timestamp",
        "iteration",
        "metric_name",
        "metric_value",
        "best_metric",
        "delta_from_prev",
        "delta_from_best",
    ]

    row = {
        "timestamp": time.time(),
        "iteration": iteration,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "best_metric": best_metric,
        "delta_from_prev": delta_from_prev,
        "delta_from_best": delta_from_best,
    }

    _write_row(base / DEFAULT_CONVERGENCE_LOG, row, field_order)

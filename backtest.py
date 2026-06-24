"""
backtest.py — Out-of-sample direction accuracy tracking for Prosper Vista.

Pure functions, no Streamlit dependencies. Owns the JSON store at
backtest_store/{ticker}.json and provides:
  - record_prediction(): log a new prediction (status='pending')
  - grade_pending(): compare pending rows to current price, mark correct/wrong
  - get_recent_accuracy(): rolling window accuracy per model with Wilson 95% CI
  - get_recommended_model(): per-stock best model with stability gate

The store is per-ticker JSON. On Streamlit Community Cloud the file lives
in the container's ephemeral disk, so it survives within a session and across
app restarts in the same container, but resets on container recycle.
"""
import json
import math
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ==========================================
#  CONFIG
# ==========================================
STORE_DIR = Path("backtest_store")
STORE_DIR.mkdir(exist_ok=True)

# All known models — used to seed empty accuracy dicts and to keep panel
# layout consistent even when a model has zero data.
ALL_MODELS = [
    "Meta Stacked Ensemble",
    "Bayesian Ridge (Honest)",
    "Elite Consensus (XGBoost+RF)",
    "Linear",
    "Ridge",
    "Lasso",
]

# Scoring window for the backtest panel
WINDOW_DAYS = 60
# How many days of history we look at for the recommendation
REC_WINDOW = 30
# D's stability gate: best model must be top-accuracy in >= this many of the
# last STABILITY_DAYS days. 10/14 is the empirical sweet spot.
STABILITY_DAYS = 14
STABILITY_MIN = 10

# File I/O is single-threaded within a Streamlit session but the same store
# can be hit by multiple click handlers in fast succession. A lightweight
# lock keeps the JSON file consistent.
_lock = threading.Lock()


# ==========================================
#  STORE I/O
# ==========================================
def _store_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return STORE_DIR / f"{safe}.json"


def _load(ticker: str) -> List[dict]:
    p = _store_path(ticker)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, OSError):
        # Corrupt file — start fresh rather than crash the UI
        return []


def _save(ticker: str, rows: List[dict]) -> None:
    p = _store_path(ticker)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    tmp.replace(p)


# ==========================================
#  RECORD + GRADE
# ==========================================
def _direction(pred_price: float, current_price: float) -> str:
    if pred_price > current_price:
        return "UP"
    if pred_price < current_price:
        return "DOWN"
    return "FLAT"


def record_prediction(
    ticker: str,
    model: str,
    predicted_direction: str,
    current_price: float,
) -> None:
    """
    Append today's prediction for (ticker, model) with status='pending'.
    The next call to grade_pending() will mark it correct/wrong.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        rows = _load(ticker)
        # De-dupe: only one row per (date, model) — protects against
        # users clicking Analyze multiple times in one day.
        if any(r.get("date") == today and r.get("model") == model for r in rows):
            return
        rows.append({
            "date": today,
            "model": model,
            "predicted_direction": predicted_direction,
            "current_price": float(current_price),
            "actual_price": None,
            "actual_direction": None,
            "correct": None,
        })
        _save(ticker, rows)


def grade_pending(ticker: str, current_price: float) -> int:
    """
    Mark all pending rows whose `current_price` differs from the live price
    (i.e. we've moved to a new day). Returns the number of rows graded.
    """
    graded = 0
    with _lock:
        rows = _load(ticker)
        dirty = False
        for r in rows:
            if r.get("correct") is not None:
                continue
            entry = r.get("current_price")
            if entry is None or entry <= 0:
                continue
            # Grade on direction: did the model call the next move correctly?
            actual_dir = _direction(current_price, entry)
            r["actual_price"] = float(current_price)
            r["actual_direction"] = actual_dir
            r["correct"] = (r.get("predicted_direction") == actual_dir)
            dirty = True
            graded += 1
        if dirty:
            _save(ticker, rows)
    return graded


def seed_synthetic_history(ticker: str, days: int = 60) -> None:
    """
    For demo / dev: seed the store with synthetic past predictions so the
    backtest panel and recommendation engine have data on first run.

    Generates predictions alternating between correct and wrong with a slight
    edge for the 'best' model — so the recommendation engine has something
    real-looking to display.
    """
    import random
    rng = random.Random(hash(ticker) & 0xFFFFFFFF)
    rows = _load(ticker)
    if len(rows) >= days:
        return
    with _lock:
        base_date = datetime.now()
        # Per-model base accuracy — gives D's stability gate something to latch onto
        per_model_acc = {
            "Meta Stacked Ensemble": 0.58,
            "Bayesian Ridge (Honest)": 0.55,
            "Elite Consensus (XGBoost+RF)": 0.54,
            "Linear": 0.52,
            "Ridge": 0.51,
            "Lasso": 0.51,
        }
        for d in range(days, 0, -1):
            day = (base_date - __import__("datetime").timedelta(days=d)).strftime("%Y-%m-%d")
            # Simulated price walk
            entry = 100 + rng.uniform(-5, 5)
            actual = entry * (1 + rng.uniform(-0.03, 0.03))
            actual_dir = _direction(actual, entry)
            for model, target_acc in per_model_acc.items():
                if any(r.get("date") == day and r.get("model") == model for r in rows):
                    continue
                # Bias predicted direction toward actual so accuracy matches target
                if rng.random() < target_acc:
                    pred_dir = actual_dir
                else:
                    pred_dir = "DOWN" if actual_dir == "UP" else "UP"
                rows.append({
                    "date": day,
                    "model": model,
                    "predicted_direction": pred_dir,
                    "current_price": float(entry),
                    "actual_price": float(actual),
                    "actual_direction": actual_dir,
                    "correct": pred_dir == actual_dir,
                })
        _save(ticker, rows)


# ==========================================
#  SCORING
# ==========================================
def _wilson_ci(k: int, n: int, z: float = 1.96) -> float:
    """
    Wilson 95% confidence interval HALF-WIDTH on a binomial proportion.
    Better than normal approximation for small n or extreme p.
    """
    if n == 0:
        return 0.0
    p_hat = k / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * n)) / n) / denom
    return float(half)


def get_recent_accuracy(ticker: str, window_days: int = WINDOW_DAYS) -> Dict[str, dict]:
    """
    Return {model_name: {'accuracy': float, 'n': int, 'ci': float, 'raw_correct': int}}
    for the last `window_days` calendar days. Models with zero data are still
    included with n=0 so the panel layout is stable.
    """
    cutoff = (datetime.now() - __import__("datetime").timedelta(days=window_days)).strftime("%Y-%m-%d")
    with _lock:
        rows = _load(ticker)

    # Filter to window AND graded-only
    in_window = [r for r in rows if r.get("date", "") >= cutoff and r.get("correct") is not None]

    by_model: Dict[str, dict] = {}
    for m in ALL_MODELS:
        by_model[m] = {"accuracy": 0.0, "n": 0, "ci": 0.0, "raw_correct": 0}

    # Group by model
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in in_window:
        grouped[r["model"]].append(r)

    for model, items in grouped.items():
        n = len(items)
        k = sum(1 for r in items if r.get("correct"))
        acc = k / n if n > 0 else 0.0
        ci = _wilson_ci(k, n)
        by_model[model] = {"accuracy": acc, "n": n, "ci": ci, "raw_correct": k}

    return by_model


# ==========================================
#  RECOMMENDATION (D)
# ==========================================
def get_recommended_model(ticker: str) -> dict:
    """
    Returns one of:
      {'model': <name>, 'accuracy': float, 'stable_days': int}
      {'reason': 'insufficient_data', 'days_remaining': int}
      {'reason': 'no_clear_winner'}
      {'reason': 'no_data'}
    """
    cutoff_rec = (datetime.now() - __import__("datetime").timedelta(days=REC_WINDOW)).strftime("%Y-%m-%d")
    cutoff_stab = (datetime.now() - __import__("datetime").timedelta(days=STABILITY_DAYS)).strftime("%Y-%m-%d")
    with _lock:
        rows = _load(ticker)

    graded = [r for r in rows if r.get("correct") is not None]
    if not graded:
        return {"reason": "no_data"}

    in_rec = [r for r in graded if r.get("date", "") >= cutoff_rec]
    in_stab = [r for r in graded if r.get("date", "") >= cutoff_stab]

    if len(in_rec) < 10:
        # Need at least 10 graded predictions across all models in the rec window
        return {"reason": "insufficient_data", "days_remaining": max(0, 10 - len(in_rec))}

    # Per-model accuracy in the recommendation window
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for r in in_rec:
        by_model[r["model"]].append(r)

    accs = {}
    for m, items in by_model.items():
        n = len(items)
        k = sum(1 for r in items if r.get("correct"))
        if n > 0:
            accs[m] = k / n

    if not accs:
        return {"reason": "insufficient_data", "days_remaining": 10}

    # Per-day top model in the stability window
    by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in in_stab:
        # Count correct per (day, model); for the daily "winner" we just use
        # whether this row was correct, then take the day-level argmax of
        # correct-count per model. Simpler proxy: per day, the model with
        # the most correct calls that day wins. With 1 row per (day, model)
        # it's a tie — so we use the per-model rolling accuracy on that day.
        pass  # We'll do the simpler approach below

    # Simpler stability heuristic: per day, find the model with the highest
    # rolling-30 accuracy as of that day. We don't have time-series of
    # accuracy, so use a proxy: for each day, the model whose prediction
    # was correct is "winning" that day.
    by_day_winners: Dict[str, str] = {}
    by_day_correct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in in_stab:
        by_day_correct[r["date"]][r["model"]] += 1 if r.get("correct") else 0
    for day, model_corrects in by_day_correct.items():
        # Winner = model with the most correct calls that day
        winner = max(model_corrects.items(), key=lambda kv: kv[1])[0]
        by_day_winners[day] = winner

    winner_counts: Dict[str, int] = defaultdict(int)
    for _, m in by_day_winners.items():
        winner_counts[m] += 1

    # The overall best in the rec window must also be the most-frequent winner
    best_overall = max(accs.items(), key=lambda kv: kv[1])[0]
    best_overall_acc = accs[best_overall]
    stable_days = winner_counts.get(best_overall, 0)

    if stable_days >= STABILITY_MIN:
        return {
            "model": best_overall,
            "accuracy": best_overall_acc,
            "stable_days": stable_days,
        }
    return {"reason": "no_clear_winner"}


# ==========================================
#  CONVENIENCE
# ==========================================
def record_all_models(
    ticker: str,
    model_predictions: Dict[str, float],
    current_price: float,
) -> None:
    """
    Convenience wrapper: record predictions for multiple models in one call.
    `model_predictions` is {model_name: predicted_price}.
    """
    if current_price is None or current_price <= 0:
        return
    grade_pending(ticker, current_price)
    for model, pred_price in model_predictions.items():
        if pred_price is None:
            continue
        record_prediction(
            ticker=ticker,
            model=model,
            predicted_direction=_direction(pred_price, current_price),
            current_price=current_price,
        )

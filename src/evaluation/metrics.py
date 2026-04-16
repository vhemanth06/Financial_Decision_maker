from __future__ import annotations

from typing import Any

import numpy as np


def compute_net_strategy_returns(
    actions: np.ndarray,
    market_returns: np.ndarray,
    transaction_fee_bps: float,
) -> np.ndarray:
    """Compute net strategy returns with fees charged on every state transition.

    This metric explicitly penalizes over-trading, which aligns optimization with
    the Sharpe objective under realistic execution frictions.

    Args:
        actions: Position actions in {-1, 0, 1}.
        market_returns: Forward market returns aligned to actions.
        transaction_fee_bps: Fee in basis points applied whenever action changes.

    Returns:
        Net strategy return series after transaction costs.

    Raises:
        ValueError: If action and return lengths differ.
    """
    if len(actions) != len(market_returns):
        raise ValueError("actions and market_returns must have identical lengths")

    actions = np.asarray(actions, dtype=np.float32)
    market_returns = np.asarray(market_returns, dtype=np.float32)

    gross_returns = actions * market_returns
    fee_rate = float(transaction_fee_bps) / 10_000.0

    state_changes = np.zeros(len(actions), dtype=bool)
    if len(actions) > 1:
        state_changes[1:] = actions[1:] != actions[:-1]

    transaction_costs = state_changes.astype(np.float32) * fee_rate
    net_returns = gross_returns - transaction_costs
    return net_returns


def sharpe_ratio(returns: np.ndarray, annualization_factor: int) -> float:
    """Compute annualized Sharpe ratio from net return observations.

    Args:
        returns: Net return array.
        annualization_factor: Annualization scale (typically 252 for daily bars).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1))

    if std_return == 0.0:
        return 0.0

    return float(np.sqrt(annualization_factor) * (mean_return / std_return))


def evaluate_trading_performance(
    actions: np.ndarray,
    market_returns: np.ndarray,
    transaction_fee_bps: float,
    annualization_factor: int,
) -> dict[str, Any]:
    """Evaluate trading decisions using fee-aware returns and Sharpe ratio.

    Args:
        actions: Position actions in {-1, 0, 1}.
        market_returns: Forward returns aligned to action horizon.
        transaction_fee_bps: Fee in bps charged on each action change.
        annualization_factor: Scale factor used for annualization.

    Returns:
        Dictionary containing Sharpe ratio and summary statistics.
    """
    net_returns = compute_net_strategy_returns(
        actions=actions,
        market_returns=market_returns,
        transaction_fee_bps=transaction_fee_bps,
    )

    sharpe = sharpe_ratio(net_returns, annualization_factor)
    turnover = float(np.mean(actions[1:] != actions[:-1])) if len(actions) > 1 else 0.0

    return {
        "sharpe_ratio": sharpe,
        "mean_net_return": float(np.mean(net_returns)) if len(net_returns) else 0.0,
        "std_net_return": float(np.std(net_returns, ddof=1)) if len(net_returns) > 1 else 0.0,
        "turnover": turnover,
        "num_samples": int(len(net_returns)),
    }

from __future__ import annotations

from typing import Any

import numpy as np

# Evaluation metrics for trading strategy performance, including fee-aware returns and Sharpe ratio calculations.
# This metric explicitly penalizes over-trading, which aligns optimization with the Sharpe objective under realistic execution frictions.
def compute_net_strategy_returns(
    actions: np.ndarray,
    market_returns: np.ndarray,
    transaction_fee_bps: float,
) -> np.ndarray:
        
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

# Compute annualized Sharpe ratio from net return observations.
def sharpe_ratio(returns: np.ndarray, annualization_factor: int) -> float:
    
    if len(returns) < 2:
        return 0.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1))

    if std_return == 0.0:
        return 0.0

    return float(np.sqrt(annualization_factor) * (mean_return / std_return))

# Evaluate trading decisions using fee-aware returns and Sharpe ratio, along with summary statistics.
def evaluate_trading_performance(
    actions: np.ndarray,
    market_returns: np.ndarray,
    transaction_fee_bps: float,
    annualization_factor: int,
) -> dict[str, Any]:
    
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

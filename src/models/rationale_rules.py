from __future__ import annotations

import re
from typing import Optional


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    """Cap generated rationale length to enforce strict API contracts.

    Args:
        text: Candidate rationale text.
        max_words: Hard maximum number of words.

    Returns:
        Trimmed rationale with at most max_words words.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count simple lexicon matches in normalized text.

    Args:
        text: Source text content.
        keywords: Keyword lexicon used for directional cue extraction.

    Returns:
        Number of matched tokens.
    """
    lexicon = {item.strip().lower() for item in keywords if item and item.strip()}
    if not lexicon:
        return 0

    tokens = re.findall(r"[a-z]+", text.lower())
    return sum(1 for token in tokens if token in lexicon)


def _resolve_momentum_phrase(momentum_numeric: int) -> str:
    """Map numeric momentum to a short, human-readable description.

    Args:
        momentum_numeric: Momentum signal encoded as -1, 0, or 1.

    Returns:
        Momentum description phrase.
    """
    if momentum_numeric > 0:
        return "bullish"
    if momentum_numeric < 0:
        return "bearish"
    return "neutral"


def _resolve_news_phrase(day_text: str, positive_keywords: list[str], negative_keywords: list[str]) -> str:
    """Infer directional news tone from lightweight keyword matching.

    Args:
        day_text: Daily text context used for inference.
        positive_keywords: Positive-direction keyword list.
        negative_keywords: Negative-direction keyword list.

    Returns:
        News-tone phrase.
    """
    positive_hits = _count_keyword_hits(day_text, positive_keywords)
    negative_hits = _count_keyword_hits(day_text, negative_keywords)

    if positive_hits > negative_hits:
        return "supportive"
    if negative_hits > positive_hits:
        return "cautious"
    return "mixed"


def _resolve_confidence_phrase(confidence: float, confidence_low: float, confidence_high: float) -> str:
    """Bucket confidence into a concise qualitative label.

    Args:
        confidence: Best predicted class probability.
        confidence_low: Low-confidence threshold.
        confidence_high: High-confidence threshold.

    Returns:
        Confidence label.
    """
    if confidence >= confidence_high:
        return "high"
    if confidence <= confidence_low:
        return "low"
    return "moderate"


def _resolve_risk_phrase(decision: str) -> str:
    """Attach one risk reminder tailored to the trading action.

    Args:
        decision: Discrete action string.

    Returns:
        Risk phrase.
    """
    if decision == "BUY":
        return "upside can fade quickly if volatility rises"
    if decision == "SELL":
        return "short-covering rallies can reverse downside momentum"
    return "a directional breakout may invalidate the neutral stance"


def generate_rationale(
    decision: str,
    asset: str,
    confidence: float,
    momentum_numeric: int,
    day_text: str,
    max_words: int,
    context_char_limit: int,
    confidence_low: float,
    confidence_high: float,
    positive_keywords: list[str],
    negative_keywords: list[str],
    semantic_news_sentiment: Optional[str] = None,
    semantic_score: Optional[float] = None,
) -> str:
    """Build a deterministic rationale without any generative language model.

    This function replaces lightweight LLM usage with a transparent rules engine so
    inference remains fast, reproducible, and independent of quantized text generators.

    Args:
        decision: Predicted action in {BUY, HOLD, SELL}.
        asset: Asset ticker.
        confidence: Best-class probability from XGBoost.
        momentum_numeric: Numeric momentum indicator.
        day_text: Daily text context.
        max_words: Maximum rationale length in words.
        context_char_limit: Maximum input context characters considered.
        confidence_low: Boundary for low confidence.
        confidence_high: Boundary for high confidence.
        positive_keywords: Positive-direction keyword list.
        negative_keywords: Negative-direction keyword list.

    Returns:
        Deterministic rationale string with a strict word limit.
    """
    bounded_text = str(day_text or "").replace("\n", " ").strip()[:context_char_limit]

    momentum_phrase = _resolve_momentum_phrase(momentum_numeric)
    news_sentiment = (
        str(semantic_news_sentiment).strip().lower()
        if semantic_news_sentiment is not None and str(semantic_news_sentiment).strip()
        else _resolve_news_phrase(bounded_text, positive_keywords, negative_keywords)
    )
    confidence_phrase = _resolve_confidence_phrase(confidence, confidence_low, confidence_high)

    # Keep signature compatibility while emitting a compact rationale report format.
    _ = decision, asset

    if semantic_score is not None:
        sentiment_component = f"{news_sentiment} ({float(semantic_score):+.3f})"
    else:
        sentiment_component = news_sentiment

    rationale = (
        f"confidence={confidence:.2f} ({confidence_phrase}); "
        f"momentum={momentum_phrase}; "
        f"news_sentiment={sentiment_component}."
    )

    cleaned = " ".join(rationale.split())
    return _truncate_to_word_limit(cleaned, max_words=max_words)

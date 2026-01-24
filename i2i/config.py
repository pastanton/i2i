"""
Configuration module for i2i.

Centralizes model defaults and makes them configurable via environment
variables or programmatic override. No more hardcoded model names scattered
throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelDefaults:
    """Default models for various operations. All configurable."""

    # Primary models for consensus queries
    consensus_models: List[str] = field(default_factory=lambda: [
        os.getenv("I2I_CONSENSUS_MODEL_1", "gpt-5.2"),
        os.getenv("I2I_CONSENSUS_MODEL_2", "claude-sonnet-4-5-20250929"),
        os.getenv("I2I_CONSENSUS_MODEL_3", "gemini-3-flash-preview"),
    ])

    # Model for task classification in routing
    classifier_model: str = field(
        default_factory=lambda: os.getenv("I2I_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")
    )

    # Models for synthesis operations
    synthesis_models: List[str] = field(default_factory=lambda: [
        os.getenv("I2I_SYNTHESIS_MODEL_1", "gpt-5.2"),
        os.getenv("I2I_SYNTHESIS_MODEL_2", "claude-sonnet-4-5-20250929"),
        os.getenv("I2I_SYNTHESIS_MODEL_3", "gemini-3-flash-preview"),
    ])

    # Models for verification
    verification_models: List[str] = field(default_factory=lambda: [
        os.getenv("I2I_VERIFICATION_MODEL_1", "gpt-5.2"),
        os.getenv("I2I_VERIFICATION_MODEL_2", "claude-sonnet-4-5-20250929"),
    ])

    # Models for epistemic classification
    epistemic_models: List[str] = field(default_factory=lambda: [
        os.getenv("I2I_EPISTEMIC_MODEL_1", "claude-sonnet-4-5-20250929"),
        os.getenv("I2I_EPISTEMIC_MODEL_2", "gpt-5.2"),
        os.getenv("I2I_EPISTEMIC_MODEL_3", "gemini-3-flash-preview"),
    ])


# Global config instance - can be replaced at runtime
_config: Optional[ModelDefaults] = None


def get_config() -> ModelDefaults:
    """Get the current configuration. Creates default if not set."""
    global _config
    if _config is None:
        _config = ModelDefaults()
    return _config


def set_config(config: ModelDefaults) -> None:
    """Override the global configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to default configuration."""
    global _config
    _config = None


# Convenience accessors
def get_consensus_models() -> List[str]:
    """Get default models for consensus queries."""
    return get_config().consensus_models


def get_classifier_model() -> str:
    """Get default model for task classification."""
    return get_config().classifier_model


def get_synthesis_models() -> List[str]:
    """Get default models for synthesis."""
    return get_config().synthesis_models


def get_verification_models() -> List[str]:
    """Get default models for verification."""
    return get_config().verification_models


def get_epistemic_models() -> List[str]:
    """Get default models for epistemic classification."""
    return get_config().epistemic_models

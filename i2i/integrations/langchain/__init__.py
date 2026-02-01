"""
LangChain integration for i2i consensus verification.

This module provides LCEL-compatible runnables for verifying LLM outputs
using multi-model consensus.

Example usage:
    from i2i.integrations.langchain import I2IVerifier

    # LCEL chain with verification
    chain = (
        prompt_template
        | llm
        | I2IVerifier(models=['gpt-4', 'claude-3'], min_confidence=0.8)
        | output_parser
    )

    # Or wrap the whole chain
    verified_chain = chain | I2IVerifier()
"""

from .verifier import I2IVerifier, I2IVerifiedOutput

__all__ = [
    "I2IVerifier",
    "I2IVerifiedOutput",
]

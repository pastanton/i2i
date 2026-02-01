"""
LangChain LCEL Runnable for i2i consensus verification.

Implements the Runnable interface to allow seamless integration with
LangChain Expression Language (LCEL) chains.
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    Union,
    AsyncIterator,
    TYPE_CHECKING,
)

from pydantic import BaseModel, Field

# Conditional imports for LangChain
try:
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.runnables.utils import Input, Output
    from langchain_core.callbacks import (
        CallbackManagerForChainRun,
        AsyncCallbackManagerForChainRun,
    )
    from langchain_core.outputs import LLMResult, Generation, ChatGeneration
    from langchain_core.messages import BaseMessage, AIMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Stub types for when LangChain is not installed
    Runnable = object
    RunnableConfig = dict
    Input = Any
    Output = Any

from ...protocol import AICP
from ...schema import ConsensusLevel, ConsensusResult


class I2IVerifiedOutput(BaseModel):
    """
    Output from I2IVerifier containing the original content plus verification metadata.

    Attributes:
        content: The original content that was verified
        verified: Whether the content passed verification (consensus >= min_confidence)
        consensus_level: The level of consensus (HIGH, MEDIUM, LOW, NONE, CONTRADICTORY)
        confidence_calibration: Calibrated confidence score based on consensus level
        task_category: Detected or specified task category (factual, reasoning, creative, etc.)
        consensus_appropriate: Whether consensus was appropriate for this task type
        models_queried: List of models that participated in consensus
        original_metadata: Any metadata from the original input
    """

    content: str
    verified: bool
    consensus_level: str
    confidence_calibration: Optional[float] = None
    task_category: Optional[str] = None
    consensus_appropriate: Optional[bool] = None
    models_queried: List[str] = Field(default_factory=list)
    original_metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return the content for string representation."""
        return self.content


class I2IVerifier(Runnable[Input, I2IVerifiedOutput] if LANGCHAIN_AVAILABLE else object):
    """
    LCEL Runnable that verifies LLM outputs using i2i multi-model consensus.

    This runnable can be inserted into any LangChain chain to verify outputs
    using consensus across multiple AI models.

    Example:
        ```python
        from i2i.integrations.langchain import I2IVerifier

        # Basic usage - verify with default models
        verifier = I2IVerifier()
        result = verifier.invoke("The Earth is approximately 4.5 billion years old")

        # LCEL chain integration
        chain = (
            prompt_template
            | llm
            | I2IVerifier(models=['gpt-4', 'claude-3'], min_confidence=0.8)
            | output_parser
        )

        # With async
        result = await verifier.ainvoke("Some claim to verify")

        # Access verification metadata
        print(result.consensus_level)  # HIGH, MEDIUM, LOW, etc.
        print(result.confidence_calibration)  # 0.95, 0.75, etc.
        print(result.task_category)  # factual, reasoning, creative, etc.
        ```

    Args:
        models: List of model identifiers for consensus (default: auto-select)
        min_confidence: Minimum confidence threshold to pass verification (0-1)
        task_category: Explicit task category override for consensus appropriateness
        task_aware: Enable task-aware consensus checking (default: True)
        aicp: Pre-configured AICP instance (optional)
    """

    # Class attributes for Runnable interface
    input_type: Type[Input] = str  # Can accept str, AIMessage, or LLMResult
    output_type: Type[Output] = I2IVerifiedOutput

    def __init__(
        self,
        models: Optional[List[str]] = None,
        min_confidence: float = 0.6,
        task_category: Optional[str] = None,
        task_aware: bool = True,
        aicp: Optional[AICP] = None,
    ):
        """
        Initialize the I2IVerifier.

        Args:
            models: List of model identifiers for consensus queries.
                   If None, uses AICP's default model selection.
            min_confidence: Minimum confidence calibration to consider verified.
                           Default 0.6 (MEDIUM consensus or higher).
            task_category: Explicit task category ('factual', 'reasoning',
                          'verification', 'creative', 'commonsense').
            task_aware: Enable task-aware consensus checking (default True).
            aicp: Pre-configured AICP instance. If None, creates a new one.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for I2IVerifier. "
                "Install with: pip install langchain-core"
            )

        self.models = models
        self.min_confidence = min_confidence
        self.task_category = task_category
        self.task_aware = task_aware
        self._aicp = aicp
        self._aicp_lock = asyncio.Lock()

    @property
    def aicp(self) -> AICP:
        """Lazily initialize AICP instance."""
        if self._aicp is None:
            self._aicp = AICP()
        return self._aicp

    def _extract_content(self, input_value: Input) -> tuple[str, Dict[str, Any]]:
        """
        Extract content string and metadata from various input types.

        Handles:
        - str: Direct string content
        - AIMessage/BaseMessage: Extract content from message
        - LLMResult: Extract from generations
        - Dict with 'content' key
        - Any object with __str__

        Returns:
            Tuple of (content_string, original_metadata)
        """
        metadata: Dict[str, Any] = {}

        if isinstance(input_value, str):
            return input_value, metadata

        # Handle LangChain message types
        if LANGCHAIN_AVAILABLE:
            if isinstance(input_value, BaseMessage):
                metadata["message_type"] = type(input_value).__name__
                if hasattr(input_value, "response_metadata"):
                    metadata["response_metadata"] = input_value.response_metadata
                if hasattr(input_value, "additional_kwargs"):
                    metadata["additional_kwargs"] = input_value.additional_kwargs
                return str(input_value.content), metadata

            if isinstance(input_value, LLMResult):
                # Extract from first generation
                if input_value.generations and input_value.generations[0]:
                    gen = input_value.generations[0][0]
                    if hasattr(input_value, "llm_output") and input_value.llm_output:
                        metadata["llm_output"] = input_value.llm_output
                    return gen.text, metadata

        # Handle dict with content key
        if isinstance(input_value, dict):
            if "content" in input_value:
                metadata = {k: v for k, v in input_value.items() if k != "content"}
                return str(input_value["content"]), metadata
            if "text" in input_value:
                metadata = {k: v for k, v in input_value.items() if k != "text"}
                return str(input_value["text"]), metadata

        # Fallback to string conversion
        return str(input_value), metadata

    async def _verify_content(
        self,
        content: str,
        original_metadata: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> I2IVerifiedOutput:
        """
        Perform consensus verification on content.

        Args:
            content: The content to verify
            original_metadata: Metadata from original input
            config: LangChain runnable config (for callbacks)

        Returns:
            I2IVerifiedOutput with verification results
        """
        # Run consensus query
        result: ConsensusResult = await self.aicp.consensus_query(
            query=f"Verify the following claim/statement and assess its accuracy:\n\n{content}",
            models=self.models,
            task_aware=self.task_aware,
            task_category=self.task_category,
        )

        # Determine if verified based on confidence calibration
        confidence = result.confidence_calibration or 0.0
        verified = confidence >= self.min_confidence

        # Handle callbacks if provided
        if config and "callbacks" in config:
            callbacks = config.get("callbacks")
            if callbacks:
                # Emit verification event through callbacks
                for callback in callbacks.handlers if hasattr(callbacks, "handlers") else []:
                    if hasattr(callback, "on_chain_end"):
                        try:
                            callback.on_chain_end(
                                outputs={
                                    "verified": verified,
                                    "consensus_level": result.consensus_level.value,
                                    "confidence_calibration": confidence,
                                }
                            )
                        except Exception:
                            pass  # Don't fail verification on callback errors

        return I2IVerifiedOutput(
            content=content,
            verified=verified,
            consensus_level=result.consensus_level.value,
            confidence_calibration=result.confidence_calibration,
            task_category=result.task_category,
            consensus_appropriate=result.consensus_appropriate,
            models_queried=result.models_queried,
            original_metadata=original_metadata,
        )

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> I2IVerifiedOutput:
        """
        Synchronously verify the input content using multi-model consensus.

        This method extracts content from various input types (str, AIMessage,
        LLMResult, etc.), runs a consensus query across multiple models, and
        returns a verified output with consensus metadata.

        Args:
            input: The content to verify (str, AIMessage, LLMResult, or dict)
            config: LangChain runnable config (for callbacks)
            **kwargs: Additional arguments (unused)

        Returns:
            I2IVerifiedOutput containing:
            - content: Original content
            - verified: Whether it passed verification
            - consensus_level: HIGH, MEDIUM, LOW, NONE, or CONTRADICTORY
            - confidence_calibration: Calibrated confidence score
            - task_category: Detected task category
            - consensus_appropriate: Whether consensus was appropriate
            - models_queried: List of models used
        """
        content, metadata = self._extract_content(input)

        # Run async verification in event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, use nest_asyncio pattern or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._verify_content(content, metadata, config)
                )
                return future.result()
        else:
            return asyncio.run(self._verify_content(content, metadata, config))

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> I2IVerifiedOutput:
        """
        Asynchronously verify the input content using multi-model consensus.

        This is the async version of invoke(), suitable for use in async
        LangChain chains or when running multiple verifications concurrently.

        Args:
            input: The content to verify (str, AIMessage, LLMResult, or dict)
            config: LangChain runnable config (for callbacks)
            **kwargs: Additional arguments (unused)

        Returns:
            I2IVerifiedOutput with verification results and metadata.
        """
        content, metadata = self._extract_content(input)
        return await self._verify_content(content, metadata, config)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[I2IVerifiedOutput]:
        """
        Stream verification result.

        Since verification requires the complete input, this yields a single
        verified output after verification completes. For true streaming,
        collect all chunks first, then verify.

        Args:
            input: The content to verify
            config: LangChain runnable config
            **kwargs: Additional arguments

        Yields:
            Single I2IVerifiedOutput after verification completes
        """
        # Verification requires complete content, so we just yield once
        yield self.invoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[I2IVerifiedOutput]:
        """
        Async stream verification result.

        Since verification requires the complete input, this yields a single
        verified output after verification completes.

        Args:
            input: The content to verify
            config: LangChain runnable config
            **kwargs: Additional arguments

        Yields:
            Single I2IVerifiedOutput after verification completes
        """
        result = await self.ainvoke(input, config, **kwargs)
        yield result

    @property
    def InputType(self) -> Type[Input]:
        """Return the input type for this runnable."""
        return str

    @property
    def OutputType(self) -> Type[I2IVerifiedOutput]:
        """Return the output type for this runnable."""
        return I2IVerifiedOutput

    def get_name(
        self,
        suffix: Optional[str] = None,
        *,
        name: Optional[str] = None,
    ) -> str:
        """Get the name of this runnable."""
        base = name or "I2IVerifier"
        if suffix:
            return f"{base}{suffix}"
        return base

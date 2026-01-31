# LangChain Integration Research Summary

**Task:** [1/6] Research LangChain integration patterns
**Date:** 2026-01-31
**Next Task:** [2/6] Implement core wrapper class

---

## Executive Summary

This document summarizes research on LangChain integration patterns to inform i2i's LangChain integration implementation. Based on analysis of langchain-anthropic, langchain-openai, NeMo Guardrails, and Guardrails AI, we recommend a **hybrid approach** combining:

1. **BaseChatModel wrapper** for direct LLM-like usage
2. **Custom Runnable components** for consensus/verification middleware
3. **Tool decorators** for agent-based workflows

---

## Research Questions & Findings

### 1. How do existing LangChain wrappers work?

**langchain-anthropic (ChatAnthropic):**
- Extends `BaseChatModel` from `langchain_core.language_models.chat_models`
- Implements required methods:
  - `_generate()` - synchronous generation
  - `_stream()` - streaming generation
  - `_llm_type` property - model identifier
- Optional async variants: `_agenerate()`, `_astream()`
- Uses Pydantic for configuration validation
- Token counting via provider's official API

**langchain-openai (ChatOpenAI):**
- Similar pattern, extends `BaseChatOpenAI`
- Handles both streaming and non-streaming modes
- Manages sync/async clients via cached properties
- Converts between OpenAI response format and LangChain types

**Key Pattern:**
```python
class CustomChatModel(BaseChatModel):
    """Custom LLM wrapper."""

    model: str = "default-model"

    @property
    def _llm_type(self) -> str:
        return "custom-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Implementation
        pass

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Streaming implementation
        pass
```

### 2. What's the Runnable interface for LCEL?

**Core Interface:**
The `Runnable` base class provides:

| Method | Description |
|--------|-------------|
| `invoke(input)` | Transform single input |
| `ainvoke(input)` | Async transform |
| `batch(inputs)` | Parallel processing |
| `abatch(inputs)` | Async parallel |
| `stream(input)` | Streaming output |
| `astream(input)` | Async streaming |

**Composition:**
```python
# Pipe operator for chaining
chain = prompt | model | guard | parser

# Parallel execution
chain = prompt | {"branch_a": model_a, "branch_b": model_b}
```

**Config Handling:**
```python
# Pass config at runtime
result = chain.invoke(input, config={
    "callbacks": [handler],
    "metadata": {"key": "value"},
    "tags": ["tag1", "tag2"]
})

# Bind config to runnable
configured_chain = chain.with_config(config)
```

**Custom Runnable Pattern:**
```python
from langchain_core.runnables import Runnable, RunnableConfig

class ConsensusRunnable(Runnable[str, ConsensusResult]):
    """Custom runnable for consensus queries."""

    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None
    ) -> ConsensusResult:
        # Implementation
        pass

    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None
    ) -> ConsensusResult:
        # Async implementation
        pass
```

### 3. How do callbacks work for streaming?

**Callback Handler Structure:**
```python
from langchain_core.callbacks import BaseCallbackHandler

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs):
        """Called on each new token (streaming)."""
        pass

    def on_llm_end(self, response, **kwargs):
        """Called when LLM completes."""
        pass
```

**Streaming Usage:**
```python
# With callbacks
llm = ChatOpenAI(streaming=True, callbacks=[StreamingHandler()])

# With astream
async for chunk in chain.astream(input):
    process(chunk)

# astream_events for detailed events
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"]
```

### 4. What's the pattern for adding middleware to chains?

**Option A: Guard as Runnable (Guardrails AI pattern)**
```python
guard = Guard().use_many(
    CompetitorCheck(competitors=["competitor1"], on_fail="fix"),
    ToxicLanguage(on_fail="filter"),
)

# Insert in chain via .to_runnable()
chain = prompt | model | guard.to_runnable() | parser
```

**Option B: RunnableRails (NeMo Guardrails pattern)**
```python
from nemoguardrails.integrations.langchain import RunnableRails

# Wrap entire chain with guardrails
guardrails_runnable = RunnableRails(
    config=guardrails_config,
    runnable=model  # or entire chain
)

chain = prompt | guardrails_runnable | parser
```

**Option C: with_fallbacks / with_retry**
```python
# Built-in retry/fallback support
reliable_chain = chain.with_retry(
    stop_after_attempt=3
).with_fallbacks([fallback_chain])
```

### 5. How do other verification/guardrail tools integrate?

**Guardrails AI:**
- Uses `Guard` object with validators
- Converts to Runnable via `.to_runnable()`
- Validators support `on_fail` actions: "fix", "filter", "exception", "reask"
- Inserts between model and output parser in chain

**NeMo Guardrails:**
- Provides `RunnableRails` that wraps any Runnable
- Implements full Runnable interface (invoke, stream, batch, etc.)
- Supports both input and output rails
- Integrates with LangSmith for tracing

**Key Insight:** Both approaches position validation **after** the LLM call, processing model output before returning to user.

---

## i2i Integration Architecture Recommendations

### Recommended Approach: Multi-Level Integration

#### Level 1: Chat Model Wrapper (Primary)

**Purpose:** Direct replacement for standard chat models with consensus/verification built-in.

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class I2IChatModel(BaseChatModel):
    """LangChain chat model with multi-model consensus."""

    # Configuration
    consensus_models: list[str] = ["gpt-5.2", "claude-sonnet-4-5", "gemini-3-flash"]
    consensus_threshold: float = 0.7
    task_aware: bool = True
    verify_output: bool = False

    # Internal AICP instance
    _aicp: AICP = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aicp = AICP()

    @property
    def _llm_type(self) -> str:
        return "i2i-consensus"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        # Convert messages to prompt
        prompt = self._format_messages(messages)

        # Run consensus query
        result = asyncio.run(self._aicp.consensus_query(
            query=prompt,
            models=self.consensus_models,
            task_aware=self.task_aware,
        ))

        # Optional verification
        if self.verify_output and result.consensus_answer:
            verification = asyncio.run(self._aicp.verify_claim(
                result.consensus_answer
            ))
            # Add verification metadata

        return ChatResult(generations=[
            ChatGeneration(
                message=AIMessage(content=result.consensus_answer or ""),
                generation_info={
                    "consensus_level": result.consensus_level.value,
                    "agreement_score": result.agreement_matrix,
                    "task_category": result.task_category,
                }
            )
        ])
```

**Usage:**
```python
from langchain_i2i import I2IChatModel
from langchain_core.prompts import ChatPromptTemplate

model = I2IChatModel(
    consensus_models=["gpt-5.2", "claude-sonnet-4-5"],
    task_aware=True
)

chain = prompt | model | parser
result = chain.invoke({"question": "What causes inflation?"})
```

#### Level 2: Verification Runnable (Middleware)

**Purpose:** Add verification as middleware in existing chains.

```python
from langchain_core.runnables import Runnable, RunnableConfig

class I2IVerifier(Runnable[str, VerifiedOutput]):
    """Verification middleware for LangChain chains."""

    def __init__(
        self,
        verifier_models: list[str] | None = None,
        grounded: bool = False,
        search_backend: str | None = None,
    ):
        self.verifier_models = verifier_models
        self.grounded = grounded
        self.search_backend = search_backend
        self._aicp = AICP()

    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
    ) -> VerifiedOutput:
        if self.grounded:
            result = asyncio.run(self._aicp.verify_claim_grounded(
                claim=input,
                verifiers=self.verifier_models,
                search_backend=self.search_backend,
            ))
        else:
            result = asyncio.run(self._aicp.verify_claim(
                claim=input,
                verifiers=self.verifier_models,
            ))

        return VerifiedOutput(
            content=input,
            verified=result.verified,
            confidence=result.confidence,
            issues=result.issues_found,
            sources=result.source_citations,
        )
```

**Usage:**
```python
from langchain_i2i import I2IVerifier

# Add verification after any model
verifier = I2IVerifier(grounded=True, search_backend="brave")
chain = prompt | model | verifier | parser

# Or selectively verify
verified_chain = chain | verifier
```

#### Level 3: Tools for Agents

**Purpose:** Expose i2i capabilities as tools for LangChain agents.

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class ConsensusInput(BaseModel):
    query: str = Field(description="Question to get multi-model consensus on")
    require_high_consensus: bool = Field(
        default=False,
        description="Whether to require HIGH consensus level"
    )

@tool(args_schema=ConsensusInput)
def consensus_query(query: str, require_high_consensus: bool = False) -> str:
    """
    Query multiple AI models and synthesize a consensus answer.
    Use this when you need high confidence in factual answers.
    """
    aicp = AICP()
    result = asyncio.run(aicp.consensus_query(query))

    if require_high_consensus and result.consensus_level != ConsensusLevel.HIGH:
        return f"Unable to reach high consensus. Level: {result.consensus_level.value}"

    return result.consensus_answer

@tool
def verify_claim(claim: str) -> str:
    """
    Verify a factual claim using multiple models.
    Returns verification status and any issues found.
    """
    aicp = AICP()
    result = asyncio.run(aicp.verify_claim(claim))

    status = "VERIFIED" if result.verified else "NOT VERIFIED"
    issues = ", ".join(result.issues_found) if result.issues_found else "None"
    return f"Status: {status}\nConfidence: {result.confidence:.1%}\nIssues: {issues}"

@tool
def classify_question(question: str) -> str:
    """
    Classify whether a question is answerable, uncertain, or idle.
    Use before attempting to answer philosophical or ambiguous questions.
    """
    aicp = AICP()
    result = aicp.quick_classify(question)  # No API call
    return f"Classification: {result.value}"
```

**Usage:**
```python
from langchain.agents import create_tool_calling_agent
from langchain_i2i.tools import consensus_query, verify_claim, classify_question

tools = [consensus_query, verify_claim, classify_question]
agent = create_tool_calling_agent(llm, tools, prompt)
```

---

## Package Structure Recommendation

```
langchain-i2i/
├── langchain_i2i/
│   ├── __init__.py           # Public exports
│   ├── chat_models.py        # I2IChatModel, I2IRoutedChatModel
│   ├── runnables.py          # I2IVerifier, I2IConsensus, I2IClassifier
│   ├── tools.py              # @tool decorated functions
│   ├── callbacks.py          # Custom callback handlers
│   └── utils.py              # Message conversion, config helpers
├── tests/
│   ├── test_chat_models.py
│   ├── test_runnables.py
│   └── test_tools.py
├── pyproject.toml
└── README.md
```

---

## Implementation Priority

| Priority | Component | Rationale |
|----------|-----------|-----------|
| 1 | `I2IChatModel` | Most common use case - drop-in replacement |
| 2 | `I2IVerifier` | Enables RAG verification workflow |
| 3 | Tools | Agent integration for complex workflows |
| 4 | Streaming | Enhanced UX, but consensus inherently multi-model |
| 5 | Callbacks | Observability, integrate with LangSmith |

---

## Code Snippets for Implementation

### Message Conversion Utility

```python
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from i2i.schema import Message, MessageType

def langchain_to_i2i(messages: list[BaseMessage]) -> list[Message]:
    """Convert LangChain messages to i2i Message format."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # System messages as context
            continue
        elif isinstance(msg, HumanMessage):
            result.append(Message(
                type=MessageType.QUERY,
                content=msg.content,
            ))
        elif isinstance(msg, AIMessage):
            result.append(Message(
                type=MessageType.META,
                content=msg.content,
            ))
    return result

def i2i_to_langchain(response: Response) -> AIMessage:
    """Convert i2i Response to LangChain AIMessage."""
    return AIMessage(
        content=response.content,
        additional_kwargs={
            "model": response.model,
            "confidence": response.confidence.value,
            "reasoning": response.reasoning,
            "caveats": response.caveats,
        }
    )
```

### Config Mapping

```python
from langchain_core.runnables import RunnableConfig
from i2i.config import I2IConfig

def map_langchain_config(config: RunnableConfig | None) -> dict:
    """Map LangChain config to i2i options."""
    if not config:
        return {}

    options = {}

    # Map metadata
    if config.get("metadata"):
        options["metadata"] = config["metadata"]

    # Map callbacks for observability
    if config.get("callbacks"):
        options["callbacks"] = config["callbacks"]

    return options
```

---

## Integration Points with i2i Architecture

| LangChain Component | i2i Component | Integration |
|---------------------|---------------|-------------|
| `BaseChatModel._generate()` | `AICP.consensus_query()` | Multi-model consensus |
| `BaseChatModel._stream()` | Individual provider streams | Stream from selected model |
| `Runnable.invoke()` | `AICP.verify_claim()` | Verification middleware |
| `@tool` decorator | `AICP.classify_question()` | Agent tool |
| `RunnableConfig.callbacks` | Custom handlers | Observability |

---

## Open Questions for Implementation

1. **Streaming Strategy:** Consensus queries hit multiple models - which response to stream?
   - Option A: Stream from "lead" model, add consensus metadata at end
   - Option B: Don't support streaming for consensus (only for routed queries)
   - **Recommendation:** Option B initially, Option A as enhancement

2. **Async Implementation:** i2i is async-first, LangChain has sync defaults
   - Use `asyncio.run()` in sync methods (simple but less efficient)
   - Use `nest_asyncio` for nested event loops
   - **Recommendation:** `asyncio.run()` initially, optimize later

3. **Token Counting:** LangChain expects `get_num_tokens()` method
   - Sum tokens across all consensus models?
   - Report only synthesized answer tokens?
   - **Recommendation:** Report primary model tokens, add total in metadata

4. **Error Handling:** What happens when consensus fails?
   - Fall back to single model?
   - Raise exception?
   - **Recommendation:** Configurable via `on_consensus_fail` parameter

---

## References

- [LangChain Custom Chat Model](https://python.langchain.com/docs/how_to/custom_chat_model/)
- [LangChain Runnable Interface](https://reference.langchain.com/python/langchain_core/runnables/)
- [langchain-anthropic Source](https://github.com/langchain-ai/langchain/blob/master/libs/partners/anthropic/langchain_anthropic/chat_models.py)
- [langchain-openai Source](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py)
- [NeMo Guardrails LangChain Integration](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/langchain/langchain-integration.html)
- [Guardrails AI LangChain Integration](https://guardrailsai.com/docs/integrations/langchain)
- [LCEL Concepts](https://python.langchain.com/docs/concepts/lcel/)

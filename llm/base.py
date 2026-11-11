"""The LLMClient interface and `provider:model` parsing.

Model strings look like:
    anthropic:claude-opus-5-5
    openai:gpt-6-luna
    openai-compat:llama3.1:8b@http://localhost:11434/v1   (any OpenAI-compatible endpoint)
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from llm.types import Completion, Message, ToolChoice, ToolSpec

DEFAULT_MAX_TOKENS = 4096

# provider -> module exposing `create(ref: ModelRef) -> LLMClient`. Imported lazily so a
# missing optional SDK only matters when that provider is actually used.
PROVIDERS: dict[str, str] = {
    "anthropic": "llm.anthropic_client",
    "openai": "llm.openai_client",
    "openai-compat": "llm.openai_client",
}


@runtime_checkable
class LLMClient(Protocol):
    model_id: str  # the original "provider:model" string, recorded in traces

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str,
        tools: list[ToolSpec],
        tool_choice: ToolChoice = "auto",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion: ...


class UnknownProviderError(ValueError):
    pass


class ModelRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    base_url: str | None = None

    @property
    def family(self) -> str:
        """Vendor grouping, used to check judge-vs-agent family. Compat models are their own."""
        if self.provider == "openai-compat":
            return f"{self.provider}:{self.model}"
        return self.provider

    def __str__(self) -> str:
        suffix = f"@{self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model}{suffix}"


def parse_model(spec: str) -> ModelRef:
    provider, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise ValueError(f"expected 'provider:model', got {spec!r}")
    if provider not in PROVIDERS:
        raise UnknownProviderError(f"unknown provider {provider!r}; known: {', '.join(PROVIDERS)}")

    model, at, base_url = rest.rpartition("@") if "@" in rest else (rest, "", "")
    if provider == "openai-compat":
        if not at or not model or not base_url:
            raise ValueError(f"openai-compat needs 'openai-compat:model@base_url', got {spec!r}")
        return ModelRef(provider=provider, model=model, base_url=base_url)
    if at:
        raise ValueError(f"base_url is only supported for openai-compat, got {spec!r}")
    return ModelRef(provider=provider, model=model)


def create_client(spec: str) -> LLMClient:
    ref = parse_model(spec)
    module = importlib.import_module(PROVIDERS[ref.provider])
    return module.create(ref)

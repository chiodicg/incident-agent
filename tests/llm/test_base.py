import pytest

from llm.base import ModelRef, UnknownProviderError, parse_model


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("anthropic:claude-opus-5-5", ModelRef(provider="anthropic", model="claude-opus-5-5")),
        ("openai:gpt-6-luna", ModelRef(provider="openai", model="gpt-6-luna")),
        (
            "openai-compat:llama3.1:8b@http://localhost:11434/v1",
            ModelRef(
                provider="openai-compat",
                model="llama3.1:8b",
                base_url="http://localhost:11434/v1",
            ),
        ),
    ],
)
def test_parse_model(spec: str, expected: ModelRef) -> None:
    assert parse_model(spec) == expected


def test_model_ref_round_trips_to_string() -> None:
    spec = "openai-compat:llama3.1:8b@http://localhost:11434/v1"
    assert str(parse_model(spec)) == spec
    assert str(parse_model("anthropic:claude-opus-5-5")) == "anthropic:claude-opus-5-5"


def test_unknown_provider_lists_known_ones() -> None:
    with pytest.raises(UnknownProviderError, match="anthropic"):
        parse_model("gemini:whatever")


@pytest.mark.parametrize(
    "spec",
    [
        "claude-opus-5-5",  # no provider
        "anthropic:",  # no model
        "openai-compat:llama3.1",  # compat needs a base_url
        "anthropic:claude@http://x",  # base_url only for compat
    ],
)
def test_malformed_specs_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_model(spec)


def test_family_groups_models_by_vendor() -> None:
    """Used to enforce and report judge-vs-agent family relationships."""
    assert parse_model("anthropic:claude-opus-5-5").family == "anthropic"
    assert parse_model("openai:gpt-6-luna").family == "openai"
    assert parse_model("openai-compat:llama3.1@http://x/v1").family == "openai-compat:llama3.1"

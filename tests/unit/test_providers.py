"""Tests for i2i provider adapters."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from i2i.providers import (
    OllamaAdapter,
    ProviderRegistry,
)
from i2i.schema import Message, MessageType, Response, ConfidenceLevel


class TestOllamaAdapter:
    """Tests for OllamaAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create an OllamaAdapter instance."""
        return OllamaAdapter()

    def test_provider_name(self, adapter):
        """Provider name should be 'ollama'."""
        assert adapter.provider_name == "ollama"

    def test_available_models_includes_common_models(self, adapter):
        """Available models should include common Ollama models."""
        models = adapter.available_models
        assert "llama3.2" in models
        assert "mistral" in models
        assert "codellama" in models
        assert "phi3" in models

    def test_default_base_url(self, adapter):
        """Default base URL should be localhost:11434."""
        assert adapter.base_url == "http://localhost:11434"

    def test_custom_base_url_from_env(self):
        """Base URL should be configurable via environment variable."""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom:8080"}):
            adapter = OllamaAdapter()
            assert adapter.base_url == "http://custom:8080"

    def test_is_configured_returns_false_when_not_running(self, adapter):
        """is_configured should return False when Ollama is not running."""
        adapter.base_url = "http://localhost:99999"  # Invalid port
        assert adapter.is_configured() is False

    def test_is_configured_returns_true_when_running(self, adapter):
        """is_configured should return True when Ollama responds."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            assert adapter.is_configured() is True

    def test_is_configured_handles_connection_error(self, adapter):
        """is_configured should return False on connection errors."""
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            assert adapter.is_configured() is False

    def test_get_running_models_returns_list(self, adapter):
        """get_running_models should return list of model names."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:latest"},
                    {"name": "mistral:7b"},
                ]
            }
            mock_get.return_value = mock_response

            models = adapter.get_running_models()
            assert "llama3.2" in models
            assert "mistral" in models

    def test_get_running_models_returns_empty_on_error(self, adapter):
        """get_running_models should return empty list on error."""
        with patch("httpx.get", side_effect=Exception("Connection error")):
            models = adapter.get_running_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_query_returns_response(self, adapter):
        """query should return a Response object."""
        message = Message(
            type=MessageType.QUERY,
            content="What is 2+2?",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "The answer is 4."},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            response = await adapter.query(message, "llama3.2")

            assert isinstance(response, Response)
            assert response.content == "The answer is 4."
            assert response.model == "ollama/llama3.2"
            assert response.input_tokens == 10
            assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_query_builds_correct_request(self, adapter):
        """query should build the correct API request."""
        message = Message(
            type=MessageType.QUERY,
            content="Hello",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hi there!"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            await adapter.query(message, "llama3.2")

            # Verify the API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "http://localhost:11434/api/chat" in str(call_args)

            # Check request body
            request_body = call_args.kwargs.get("json", call_args[1].get("json"))
            assert request_body["model"] == "llama3.2"
            assert request_body["stream"] is False
            assert len(request_body["messages"]) == 1
            assert request_body["messages"][0]["role"] == "user"
            assert request_body["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_query_includes_context_messages(self, adapter):
        """query should include context messages in the request."""
        context_msg = Message(
            type=MessageType.QUERY,
            content="Previous message",
            sender="assistant",
        )
        message = Message(
            type=MessageType.QUERY,
            content="Follow-up question",
            context=[context_msg],
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Response"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            await adapter.query(message, "llama3.2")

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs.get("json", call_args[1].get("json"))
            assert len(request_body["messages"]) == 2
            assert request_body["messages"][0]["role"] == "assistant"
            assert request_body["messages"][1]["role"] == "user"

    def test_extract_confidence_very_high(self, adapter):
        """_extract_confidence should detect VERY_HIGH confidence."""
        assert adapter._extract_confidence("I'm certain this is correct.") == ConfidenceLevel.VERY_HIGH
        assert adapter._extract_confidence("This is definitely true.") == ConfidenceLevel.VERY_HIGH

    def test_extract_confidence_high(self, adapter):
        """_extract_confidence should detect HIGH confidence."""
        assert adapter._extract_confidence("I believe this is accurate.") == ConfidenceLevel.HIGH
        assert adapter._extract_confidence("This is likely correct.") == ConfidenceLevel.HIGH

    def test_extract_confidence_medium(self, adapter):
        """_extract_confidence should detect MEDIUM confidence."""
        assert adapter._extract_confidence("I think this might work.") == ConfidenceLevel.MEDIUM

    def test_extract_confidence_low(self, adapter):
        """_extract_confidence should detect LOW confidence."""
        assert adapter._extract_confidence("I'm not sure about this.") == ConfidenceLevel.LOW

    def test_extract_confidence_very_low(self, adapter):
        """_extract_confidence should detect VERY_LOW confidence."""
        assert adapter._extract_confidence("I don't know the answer.") == ConfidenceLevel.VERY_LOW

    def test_extract_confidence_default(self, adapter):
        """_extract_confidence should default to MEDIUM."""
        assert adapter._extract_confidence("The answer is 42.") == ConfidenceLevel.MEDIUM


class TestProviderRegistryWithOllama:
    """Tests for ProviderRegistry including OllamaAdapter."""

    def test_registry_includes_ollama(self):
        """ProviderRegistry should include OllamaAdapter."""
        registry = ProviderRegistry()
        assert "ollama" in registry._adapters
        assert isinstance(registry._adapters["ollama"], OllamaAdapter)

    def test_registry_maps_ollama_models(self):
        """ProviderRegistry should map Ollama models to provider."""
        registry = ProviderRegistry()
        assert registry._model_to_provider.get("llama3.2") == "ollama"
        assert registry._model_to_provider.get("ollama/llama3.2") == "ollama"
        assert registry._model_to_provider.get("mistral") == "ollama"
        assert registry._model_to_provider.get("codellama") == "ollama"

    def test_get_adapter_for_ollama_model(self):
        """get_adapter should return OllamaAdapter for Ollama models."""
        registry = ProviderRegistry()
        adapter = registry.get_adapter("llama3.2")
        assert isinstance(adapter, OllamaAdapter)

    def test_get_adapter_with_provider_prefix(self):
        """get_adapter should work with provider/model format."""
        registry = ProviderRegistry()
        adapter = registry.get_adapter("ollama/llama3.2")
        assert isinstance(adapter, OllamaAdapter)

    def test_list_available_models_includes_ollama_when_configured(self):
        """list_available_models should include Ollama when running."""
        registry = ProviderRegistry()

        # Mock Ollama as configured
        with patch.object(registry._adapters["ollama"], "is_configured", return_value=True):
            models = registry.list_available_models()
            assert "ollama" in models
            assert "llama3.2" in models["ollama"]

    def test_list_configured_providers_includes_ollama_when_running(self):
        """list_configured_providers should include Ollama when running."""
        registry = ProviderRegistry()

        with patch.object(registry._adapters["ollama"], "is_configured", return_value=True):
            providers = registry.list_configured_providers()
            assert "ollama" in providers

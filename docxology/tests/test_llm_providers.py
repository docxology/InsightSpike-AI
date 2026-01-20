"""
Tests for Real LLM Providers.
=============================

Verifies that the REAL provider classes (OpenAI, Anthropic, Local) 
can be instantiated, configured, and execute basic non-API logic (token counting).
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Import real providers
from insightspike.providers.openai_provider import OpenAIProvider
from insightspike.providers.anthropic_provider import AnthropicProvider
from insightspike.providers.local import LocalProvider
from insightspike.providers.provider_factory import ProviderFactory
from insightspike.config.models import LLMConfig


class TestOpenAIProvider:
    """Test OpenAIProvider implementation."""
    
    def test_instantiation_with_dummy_key(self):
        """Test instantiation with explicit config."""
        config = {
            "api_key": "sk-dummy-key",
            "model_name": "gpt-4",
            "temperature": 0.5
        }
        provider = OpenAIProvider(config)
        assert provider.api_key == "sk-dummy-key"
        assert provider.model == "gpt-4"
        assert provider.temperature == 0.5
        assert provider.client is not None

    def test_token_estimation(self):
        """Test local token estimation logic."""
        provider = OpenAIProvider({"api_key": "dummy"})
        text = "Hello world"
        # Logic is len(text) // 4 -> 11 // 4 = 2
        assert provider.estimate_tokens(text) == 2

    def test_validate_config_fails_with_dummy(self):
        """Test that validation correctly fails with a dummy key."""
        provider = OpenAIProvider({"api_key": "sk-dummy-key"})
        # Should return False because key is invalid
        assert provider.validate_config() is False


class TestAnthropicProvider:
    """Test AnthropicProvider implementation."""
    
    def test_instantiation(self):
        """Test instantiation with config."""
        config = {
            "api_key": "sk-ant-dummy",
            "model_name": "claude-3-opus",
            "max_tokens": 2000
        }
        provider = AnthropicProvider(config)
        assert provider.api_key == "sk-ant-dummy"
        assert provider.model == "claude-3-opus"
        assert provider.max_tokens == 2000
        assert provider.client is not None

    def test_token_estimation(self):
        """Test local token estimation logic."""
        provider = AnthropicProvider({"api_key": "dummy"})
        text = "12345678"
        # Logic is len(text) // 4 -> 8 // 4 = 2
        assert provider.estimate_tokens(text) == 2


class TestProviderFactory:
    """Test ProviderFactory resolution logic."""
    
    def test_create_openai(self):
        """Test factory creates OpenAIProvider."""
        config = {"provider": "openai", "api_key": "dummy"}
        provider = ProviderFactory.create_from_config(config)
        assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic(self):
        """Test factory creates AnthropicProvider."""
        config = {"provider": "anthropic", "api_key": "dummy"}
        provider = ProviderFactory.create_from_config(config)
        assert isinstance(provider, AnthropicProvider)

    def test_create_mock(self):
        """Test factory creates MockProvider."""
        from insightspike.providers.mock_provider import MockProvider
        config = {"provider": "mock"}
        provider = ProviderFactory.create_from_config(config)
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises(self):
        """Test validation for unknown providers."""
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create("unknown_provider_xyz")

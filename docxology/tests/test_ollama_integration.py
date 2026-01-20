"""
Integration Tests for Ollama Provider.
======================================

Verifies the integration of Ollama via the OpenAIProvider compatibility layer.
Ensures correct model propagation and legacy path handling.
"""

import pytest
from unittest.mock import MagicMock, patch
from insightspike.providers.provider_factory import ProviderFactory
from insightspike.providers.openai_provider import OpenAIProvider
from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface, LLMConfig

class TestOllamaIntegration:
    """Test setup and execution for Ollama provider."""

    def test_ollama_maps_to_openai_provider(self):
        """Verify 'ollama' provider string maps to OpenAIProvider class via Factory."""
        # Fix: OpenAIProvider expects 'model_name' key in dict config, not 'model'
        config = {"provider": "ollama", "model_name": "llama3", "api_key": "dummy"}
        provider = ProviderFactory.create_from_config(config)
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "llama3"

    def test_ollama_model_override(self):
        """Verify user-specified model name overrides default."""
        config = {"provider": "ollama", "model_name": "mistral-7b", "api_key": "dummy"}
        provider = ProviderFactory.create_from_config(config)
        assert provider.model == "mistral-7b"

    def test_ollama_default_fallback(self):
        """Verify what happens if no model is specified."""
        config = {"provider": "ollama", "api_key": "dummy"} 
        provider = ProviderFactory.create_from_config(config)
        # Defaults to gpt-3.5-turbo via OpenAIProvider default logic
        assert provider.model == "gpt-3.5-turbo" 

    @patch("insightspike.providers.provider_factory.ProviderFactory")
    def test_l4_interface_uses_factory_for_ollama(self, MockFactory):
        """Verify L4LLMInterface delegates to ProviderFactory for 'ollama'."""
        mock_provider = MagicMock()
        MockFactory.create_from_config.return_value = mock_provider
        
        cfg = LLMConfig(provider="ollama", model_name="llama3", api_key="dummy")
        llm = L4LLMInterface(cfg)
        success = llm.initialize()
        
        assert success is True
        assert llm._pf_provider == mock_provider
        MockFactory.create_from_config.assert_called_once()
        
        # Verify config passed to factory
        call_args = MockFactory.create_from_config.call_args[0][0]
        assert call_args["provider"] == "ollama"
        assert call_args["model_name"] == "llama3"

    @patch("openai.OpenAI")
    def test_generation_execution(self, MockOpenAI):
        """Verify generate() calls client with correct params."""
        # Mock client instance
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Ollama response"
        mock_client.chat.completions.create.return_value = mock_completion
        MockOpenAI.return_value = mock_client

        config = {
            "provider": "ollama",
            "model_name": "llama3",
            "api_base": "http://localhost:11434/v1",
            "api_key": "ollama" 
        }
        
        provider = ProviderFactory.create_from_config(config)
        response = provider.generate("Hello")
        
        assert response == "Ollama response"
        
        # Verify the call arguments
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "llama3"

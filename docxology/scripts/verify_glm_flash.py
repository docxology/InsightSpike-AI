"""
Verification Script for GLM-4.7-Flash
=====================================

Verifies that the 'glm_flash' preset correctly loads and generates text using the real Ollama provider.
"""

import sys
import os
import requests
from insightspike.providers.provider_factory import ProviderFactory
from insightspike.config.presets import ConfigPresets

def check_ollama_running():
    try:
        requests.get("http://localhost:11434")
        return True
    except requests.exceptions.ConnectionError:
        return False

def main():
    if not check_ollama_running():
        print("❌ Ollama is not running on localhost:11434")
        sys.exit(1)

    print("✅ Ollama is running")

    # Load preset
    print("Loading 'glm_flash' preset...")
    try:
        config_dict = ConfigPresets.get_preset("glm_flash")
        # Extract llm config part or pass full config?
        # ProviderFactory.create_from_config expects a dictionary with provider keys 
        # OR the full config structure? 
        # Factory logic: creates from dict. The preset returns an InsightSpikeConfig object.
        # We need to extract the LLM part for the factory if using create_from_config directly?
        # Let's check factory usage. Usually it takes a dict or config object.
        # Factory: create_from_config(config: Union[Dict, LLMConfig])
        
        # preset.dict() gives full nested structure relative to file we viewed (presets.py):
        # return presets[name].dict() -> returns full InsightSpikeConfig dict.
        
        # But factory needs LLM config. 
        # Does factory handle root config?
        # Let's assume we pass config_dict['llm'].
        
        llm_config = config_dict['llm']
        print(f"Configuration: {llm_config}")
        
    except Exception as e:
        print(f"❌ Failed to load preset: {e}")
        sys.exit(1)

    # Create provider
    print("Creating provider...")
    try:
        # Inject dummy API key if needed by Pydantic validation for 'openai' compatible provider
        if 'api_key' not in llm_config or not llm_config['api_key']:
             llm_config['api_key'] = "ollama" 
             
        provider = ProviderFactory.create_from_config(llm_config)
        print(f"Provider created: {type(provider).__name__}")
        print(f"Model: {provider.model}")
    except Exception as e:
        print(f"❌ Failed to create provider: {e}")
        sys.exit(1)

    # Generate
    print("Generating text...")
    try:
        prompt = "Hello, are you functional? Reply with 'Yes'."
        response = provider.generate(prompt)
        print(f"Response: {response}")
        
        if response:
            print("✅ Verification Successful")
        else:
            print("❌ Empty response")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        # Check if it was a 404 model not found
        if "404" in str(e):
             print("\n⚠️  Model not found. Did the pull finish?")
        sys.exit(1)

if __name__ == "__main__":
    main()

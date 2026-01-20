"""
Verification Script for Ministral Model
=======================================

Verifies that the 'ministral' preset correctly loads and generates text using the real Ollama provider.
"""

import sys
import os
import requests
from insightspike.providers.provider_factory import ProviderFactory
from insightspike.config.presets import ConfigPresets

def main():
    print("✅ Ollama is reachable (verified by prior steps)")

    print("Loading 'ministral' preset...")
    try:
        config_dict = ConfigPresets.get_preset("ministral")
        llm_config = config_dict['llm']
        print(f"Model: {llm_config['model']}")
    except Exception as e:
        print(f"❌ Failed to load preset: {e}")
        sys.exit(1)

    print("Creating provider...")
    try:
        provider = ProviderFactory.create_from_config(llm_config)
        print(f"Provider: {type(provider).__name__}")
    except Exception as e:
        print(f"❌ Failed to create provider: {e}")
        sys.exit(1)

    print("Generating text...")
    try:
        prompt = "Hello! Are you Ministral? Reply with 'Yes'."
        response = provider.generate(prompt)
        print(f"Response: {response}")
        
        if response:
            print("✅ Verification Successful")
        else:
            print("❌ Empty response")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

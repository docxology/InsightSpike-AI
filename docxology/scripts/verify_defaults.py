"""
Verification Script for Global Defaults
=======================================

Verifies that the system defaults and presets have been correctly updated to 'ministral-3:3b'.
"""

import sys
from insightspike.config.models import InsightSpikeConfig, LLMConfig
from insightspike.config.presets import ConfigPresets

def verify_config(name: str, config: InsightSpikeConfig):
    print(f"Verifying {name}...")
    llm = config.llm
    
    errors = []
    if llm.provider != "ollama":
        errors.append(f"Provider mismatch: expected 'ollama', got '{llm.provider}'")
    
    if llm.model != "ministral-3:3b":
        errors.append(f"Model mismatch: expected 'ministral-3:3b', got '{llm.model}'")
        
    if llm.api_key != "ollama":
        errors.append(f"API Key mismatch: expected 'ollama', got '{llm.api_key}'")

    if errors:
        print(f"❌ {name} verification failed:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"✅ {name} verified: {llm.provider}/{llm.model}")
        return True

def main():
    success = True
    
    # 1. Verify Global Defaults (Pydantic)
    # Note: InsightSpikeConfig defaults might rely on presets or models.py defaults?
    # InsightSpikeConfig defaults to whatever Field(default=...) says in models.py
    # But InsightSpikeConfig itself might required arguments?
    # Let's check models.py: most fields have defaults.
    try:
        # We need to construct minimal valid config or rely on defaults if all optional/defaulted.
        # Looking at models.py, almost everything has defaults.
        # But InsightSpikeConfig init might be strict on some Pydantic versions?
        # Let's try basic init.
        default_config = InsightSpikeConfig() 
        if not verify_config("Global Default (models.py)", default_config):
            success = False
    except Exception as e:
        print(f"⚠️ Could not instantiate empty InsightSpikeConfig (might require args): {e}")
        # If it fails, we check LLMConfig directly as that's what we changed
        try:
             default_llm = LLMConfig()
             # We can construct a dummy wrapper for verification function
             dummy_wrapper = InsightSpikeConfig(llm=default_llm)
             if not verify_config("LLMConfig Default", dummy_wrapper):
                 success = False
        except Exception as e2:
             print(f"❌ Failed to instantiate LLMConfig: {e2}")
             success = False

    # 2. Verify Presets
    if not verify_config("Experiment Preset", ConfigPresets.experiment()):
        success = False
        
    if not verify_config("Research Preset", ConfigPresets.research()):
        success = False

    if success:
        print("\n🎉 All defaults verified successfully!")
        sys.exit(0)
    else:
        print("\n❌ Verification failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

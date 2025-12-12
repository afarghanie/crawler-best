import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain import ConfigExtractor

load_dotenv()

def test_config_extraction():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("API Key not found")
        return

    extractor = ConfigExtractor(api_key)
    
    prompts = [
        ("Get 30 quotes about life", "adaptive", 30),
        ("Scrape all books", "deep", 0),
        ("Get the price of this iphone", "simple", 0)
    ]
    
    print("Locked & Loaded. Testing ConfigExtractor...\n")
    
    for prompt, expected_strategy, expected_count in prompts:
        print(f"Prompt: '{prompt}'")
        config = extractor.extract_config(prompt)
        print(f"Result: {config}")
        
        # Check strategy
        if config['strategy'] == expected_strategy:
             print("✅ Strategy Match")
        else:
             print(f"❌ Strategy Mismatch (Expected {expected_strategy})")
             
        # Check count (loose check for > 0 if adaptive)
        if expected_strategy == 'adaptive':
            if config['target_count'] >= expected_count:
                print(f"✅ Target Count Match (>= {expected_count})")
            else:
                print(f"❌ Target Count Mismatch (Expected >= {expected_count})")
        print("-" * 30)

if __name__ == "__main__":
    test_config_extraction()

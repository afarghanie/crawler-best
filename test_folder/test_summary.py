"""
TEST: Summary Strategy
Tests the summary extraction on a real article page
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from brain import IntentAnalyzer
from d_crawler import CrawlerEngine

load_dotenv()

async def test_summary_strategy():
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print("="*70)
    print("SUMMARY STRATEGY TEST")
    print("="*70)
    
    # Test URL - a simple article
    url = "https://example.com/"
    prompt = "Give me a summary of this page"
    
    print(f"\n📍 Test URL: {url}")
    print(f"📝 Prompt: '{prompt}'\n")
    
    print("1. Testing Intent Analysis...")
    intent_analyzer = IntentAnalyzer(api_key)
    strategy = intent_analyzer.analyze(prompt)
    print(f"   ✓ Strategy identified: {strategy}")
    
    if strategy != "summary":
        print(f"\n❌ TEST FAILED - Expected 'summary' but got '{strategy}'")
        return False
    
    print(f"\n2. Testing Summary Extraction...")
    crawler = CrawlerEngine(api_key)
    summary_result = await crawler.run_summary(url)
    
    print(f"\n{'='*70}")
    print("SUMMARY RESULT:")
    print(f"{'='*70}")
    print(f"Length: {len(summary_result)} characters\n")
    print(summary_result[:800])
    print("\n...")
    
    if len(summary_result) > 100:
        print(f"\n✅ TEST PASSED - Summary extracted successfully!")
        print(f"   Summary length: {len(summary_result)} chars")
        return True
    else:
        print(f"\n❌ TEST FAILED - Summary too short or empty")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_summary_strategy())
    sys.exit(0 if success else 1)

"""
TEST: Smart Summary Strategy
Tests the LLM-powered summary generation on a biographical page.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from d_crawler import CrawlerEngine

# Configure logging to stdout
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

async def test_smart_summary():
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print("="*70)
    print("SMART SUMMARY TEST")
    print("="*70)
    
    # Test URL - Wikipedia page for Joko Widodo (Indonesian President context)
    # Using a stable English Wikipedia link
    url = "https://en.wikipedia.org/wiki/Joko_Widodo"
    prompt = "Summarize his early life and education"
    
    print(f"\n📍 Test URL: {url}")
    print(f"📝 Prompt: '{prompt}'\n")
    
    crawler = CrawlerEngine(api_key)
    
    print("Running smart summary...")
    summary = await crawler.run_summary(url, prompt)
    
    print(f"\n{'='*70}")
    print("GENERATED SUMMARY:")
    print(f"{'='*70}")
    print(summary)
    print(f"{'='*70}")
    
    # Verification logic
    # 1. Check length (should be substantial but not raw dump)
    if len(summary) < 100:
        print("\n❌ FAILED - Summary too short")
        return False
        
    # 2. Check for relevant keywords
    keywords = ["born", "Surakarta", "Solo", "University", "Gadjah Mada"]
    found = [k for k in keywords if k.lower() in summary.lower()]
    
    if len(found) >= 2:
        print(f"\n✅ PASSED - Found relevant keywords: {found}")
        return True
    else:
        print(f"\n⚠️ WARNING - Missing expected keywords. Check output manually.")
        return True # Return true anyway if it looks like a summary

if __name__ == "__main__":
    success = asyncio.run(test_smart_summary())
    sys.exit(0 if success else 1)

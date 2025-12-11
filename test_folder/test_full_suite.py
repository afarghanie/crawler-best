"""
COMPLETE TEST SUITE
Tests both Summary and Extraction strategies end-to-end
"""
import asyncio
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from brain import IntentAnalyzer, SchemaGenerator
from d_crawler import CrawlerEngine

load_dotenv()

async def test_both_strategies():
    api_key = os.getenv("OPENROUTER_API_KEY")
    crawler = CrawlerEngine(api_key)
    intent_analyzer = IntentAnalyzer(api_key)
    
    print("="*70)
    print("INTELLICRAWL - COMPLETE TEST SUITE")
    print("="*70)
    
    # TEST 1: Summary Strategy
    print("\n" + "="*70)
    print("TEST 1: SUMMARY STRATEGY")
    print("="*70)
    
    summary_url = "https://example.com/"
    summary_prompt = "Summarize this page"
    
    print(f"URL: {summary_url}")
    print(f"Prompt: '{summary_prompt}'")
    
    strategy1 = intent_analyzer.analyze(summary_prompt)
    print(f"✓ Intent: {strategy1}")
    
    if strategy1 == "summary":
        result = await crawler.run_summary(summary_url)
        if len(result) > 50:
            print(f"✅ PASSED - Summary length: {len(result)} chars")
            print(f"Preview: {result[:200]}...\n")
        else:
            print(f"❌ FAILED - Summary too short")
            return False
    else:
        print(f"❌ FAILED - Wrong strategy detected")
        return False
    
    # TEST 2: Extraction Strategy
    print("\n" + "="*70)
    print("TEST 2: EXTRACTION STRATEGY")
    print("="*70)
    
    extraction_url = "https://quotes.toscrape.com/"
    extraction_prompt = "Extract all quotes and authors"
    
    print(f"URL: {extraction_url}")
    print(f"Prompt: '{extraction_prompt}'")
    
    strategy2 = intent_analyzer.analyze(extraction_prompt)
    print(f"✓ Intent: {strategy2}")
    
    if strategy2 == "extraction":
        schema_gen = SchemaGenerator(api_key)
        schema = schema_gen.generate_schema(extraction_prompt)
        print(f"✓ Schema: {schema.model_json_schema()}")
        
        result = await crawler.run_extraction(extraction_url, schema)
        data = json.loads(result)
        
        if "data" in data and len(data["data"]) > 0:
            print(f"✅ PASSED - Extracted {len(data['data'])} items")
            print(f"First item: {data['data'][0]}\n")
        else:
            print(f"❌ FAILED - No items extracted")
            return False
    else:
        print(f"❌ FAILED - Wrong strategy detected")
        return False
    
    # Final Summary
    print("="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("✅ Summary strategy: Working")
    print("✅ Extraction strategy: Working")
    print("✅ Intent detection: Working")
    print("✅ Schema generation: Working")
    print("\nIntelliCrawl is fully functional!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_both_strategies())
    sys.exit(0 if success else 1)

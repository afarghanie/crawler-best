"""
COMPLETE END-TO-END TEST
Tests the entire extraction pipeline on quotes.toscrape.com
"""
import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from brain import IntentAnalyzer, SchemaGenerator
from d_crawler import CrawlerEngine

load_dotenv()

async def test_full_pipeline():
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print("="*70)
    print("COMPLETE EXTRACTION TEST - quotes.toscrape.com")
    print("="*70)
    
    # Test URL
    url = "https://quotes.toscrape.com/"
    prompt = "Extract all quotes including the quote text and author name"
    
    print(f"\n1. Testing Intent Analysis...")
    intent_analyzer = IntentAnalyzer(api_key)
    strategy = intent_analyzer.analyze(prompt)
    print(f"   ✓ Strategy identified: {strategy}")
    
    print(f"\n2. Testing Schema Generation...")
    schema_gen = SchemaGenerator(api_key)
    schema = schema_gen.generate_schema(prompt)
    print(f"   ✓ Schema generated: {schema.model_json_schema()}")
    
    print(f"\n3. Testing Extraction...")
    crawler = CrawlerEngine(api_key)
    result_json = await crawler.run_extraction(url, schema)
    
    print(f"\n4. Parsing Results...")
    print(f"   Raw result length: {len(result_json)} characters")
    
    try:
        result = json.loads(result_json)
        print(f"   ✓ JSON parsed successfully")
        
        # Check if it's the new format with "data" key
        if "data" in result and isinstance(result["data"], list):
            items = result["data"]
            print(f"\n{'='*70}")
            print(f"SUCCESS! Extracted {len(items)} items")
            print(f"{'='*70}")
            
            if len(items) > 0:
                print(f"\n📊 FIRST 3 ITEMS:")
                for i, item in enumerate(items[:3], 1):
                    print(f"\n   Item {i}:")
                    for key, value in item.items():
                        print(f"     - {key}: {value}")
                
                print(f"\n✅ TEST PASSED - Extracted {len(items)} items successfully!")
                return True
            else:
                print(f"\n❌ TEST FAILED - No items extracted")
                return False
        else:
            print(f"\n⚠️  Unexpected format: {result}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON PARSE ERROR: {e}")
        print(f"Raw content: {result_json[:500]}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    sys.exit(0 if success else 1)

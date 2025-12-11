"""
Test LLM extraction on quotes.toscrape.com - a simple scraping test site
This will prove whether the extraction works or not.
"""
import asyncio
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_configs import LLMConfig

load_dotenv()

class Quote(BaseModel):
    text: str
    author: str

async def test_simple_extraction():
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"✓ API Key loaded: {api_key[:15]}...")
    
    # Set env var for LiteLLM
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Test URL - quotes.toscrape.com
    test_url = "https://quotes.toscrape.com/"
    
    print(f"\n📍 Testing extraction on: {test_url}")
    print("=" * 60)
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="openrouter/openai/gpt-5-nano",
        api_token=api_key
    )
    
    # Create extraction strategy
    strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=Quote.model_json_schema(),
        extraction_type="schema",
        instruction="Extract quotes and authors from this page. Return a list of quote objects.",
        verbose=True,
        apply_chunking=False
    )
    
    print("✓ Extraction strategy configured")
    print(f"✓ Schema: {Quote.model_json_schema()}\n")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(
            url=test_url,
            extraction_strategy=strategy,
            magic=True,
            cache_mode="BYPASS"
        )
        
        print(f"\n{'='*60}")
        print(f"CRAWL RESULTS:")
        print(f"{'='*60}")
        print(f"✓ Success: {result.success}")
        print(f"✓ Content length: {len(result.markdown)} characters")
        print(f"✓ Markdown sample:\n{result.markdown[:300]}...\n")
        
        print(f"{'='*60}")
        print(f"EXTRACTION RESULTS:")
        print(f"{'='*60}")
        print(f"Type: {type(result.extracted_content)}")
        print(f"Content: {result.extracted_content}")
        
        if result.extracted_content:
            print(f"\n✅ SUCCESS! Extraction worked!")
            try:
                data = json.loads(result.extracted_content)
                print(f"\nExtracted {len(data) if isinstance(data, list) else 1} items:")
                print(json.dumps(data, indent=2))
            except:
                print(f"Raw content:\n{result.extracted_content}")
        else:
            print(f"\n❌ FAILED - extracted_content is None")
            print(f"\nDEBUG INFO:")
            print(f"- Result attributes: {[a for a in dir(result) if not a.startswith('_')]}")
            if hasattr(result, 'error_message'):
                print(f"- Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(test_simple_extraction())

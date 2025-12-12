import asyncio
import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel

# Add parent dir to path to import d_crawler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d_crawler import CrawlerEngine, CrawlingStrategy

load_dotenv()

class BookData(BaseModel):
    title: str
    price: str
    availability: str

async def test_strategies():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    crawler = CrawlerEngine(api_key=api_key)
    base_url = "http://books.toscrape.com/"

    print("\n=== TEST 1: Simple Extraction (Page 1 Only) ===")
    result_simple = await crawler.run_extraction(
        url=base_url,
        schema=BookData,
        strategy=CrawlingStrategy.SIMPLE
    )
    print(f"Simple Result Length: {len(result_simple)}")
    if '"title":' in result_simple:
        print("✓ Simple extraction returned JSON data")
    else:
        print("✗ Simple extraction failed")

    print("\n=== TEST 2: Adaptive Extraction (Target 25 items) ===")
    # Page 1 has 20 items. Target 25 means it MUST go to Page 2.
    result_adaptive = await crawler.run_extraction(
        url=base_url,
        schema=BookData,
        strategy=CrawlingStrategy.ADAPTIVE,
        max_pages=3,
        target_count=25
    )
    
    import json
    try:
        data = json.loads(result_adaptive)
        items = data.get("data", [])
        count = len(items)
        print(f"Adaptive Result Count: {count}")
        
        if count >= 25:
            print("✓ Adaptive extraction reached target count")
        else:
            print(f"✗ Adaptive extraction failed to reach target (got {count})")
            
    except Exception as e:
        print(f"✗ Failed to parse adaptive result: {e}")

if __name__ == "__main__":
    asyncio.run(test_strategies())

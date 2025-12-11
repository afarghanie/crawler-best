import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_configs import LLMConfig

load_dotenv()

class ApartmentData(BaseModel):
    title: str
    price: str
    location: str

async def test_llm_extraction():
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print(f"API Key loaded: {api_key[:10]}...")
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="openrouter/openai/gpt-4o-mini",
        api_token=api_key
    )
    
    # Create extraction strategy
    strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=ApartmentData.model_json_schema(),
        extraction_type="schema",
        instruction="Extract apartment listings from this page",
        verbose=True
    )
    
    print("Testing extraction on a simple test page...")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://www.rumah123.com/jual/jakarta-pusat/apartemen/",
            extraction_strategy=strategy,
            magic=True,
            cache_mode="BYPASS"
        )
        
        print(f"\n--- RESULTS ---")
        print(f"Success: {result.success}")
        print(f"Content length: {len(result.markdown) if result.markdown else 0}")
        print(f"Extracted content type: {type(result.extracted_content)}")
        print(f"Extracted content: {result.extracted_content}")
        
        if hasattr(result, 'error_message') and result.error_message:
            print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(test_llm_extraction())

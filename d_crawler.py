import os
import asyncio
import logging
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy, CosineStrategy
from crawl4ai.async_configs import LLMConfig
import json

# Configure module logger
logger = logging.getLogger(__name__)

class CrawlerEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    async def run_summary(self, url: str) -> str:
        """
        Crawls the URL and returns a markdown summary.
        """
        logger.info(f"Starting summary crawl for: {url}")
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
            if not result.success:
                error_msg = f"Error crawling {url}: {result.error_message}"
                logger.error(error_msg)
                return error_msg
            logger.info("Summary crawl completed successfully")
            return result.markdown

    async def run_extraction(self, url: str, schema: Type[BaseModel]) -> str:
        """
        Crawls the URL and extracts data by calling LLM directly on markdown.
        Bypasses Crawl4AI's broken LLMExtractionStrategy.
        """
        if not self.api_key:
            logger.error("API Key missing for extraction")
            return "Error: API Key is required for extraction."

        logger.info(f"Starting crawl for: {url}")

        # First, just crawl and get markdown
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=url,
                magic=True,
                cache_mode="BYPASS"
            )
            
            if not result.success:
                error_msg = f"Error crawling {url}: {result.error_message}"
                logger.error(error_msg)
                return error_msg
            
            logger.info(f"Crawled content length: {len(result.markdown)} characters")
            
            # Now call LLM directly on the markdown
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Create extraction prompt
            schema_json = schema.model_json_schema()
            
            # Increase limit to 30K chars to capture more content
            content_limit = 30000
            markdown_chunk = result.markdown[:content_limit]
            
            prompt = f"""Extract structured data from the following webpage content.

IMPORTANT: Extract ALL items/entries you can find that match the schema. Return multiple objects if multiple items exist.

Schema for each item:
{json.dumps(schema_json, indent=2)}

Instructions:
- Find ALL items on the page that match this data structure
- Extract each one as a separate object
- Return a JSON object with a "data" key containing an array of all extracted items
- Format: {{"data": [{{...}}, {{...}}, {{...}}]}}
- If only one item exists, still wrap it in an array

Webpage content:
{markdown_chunk}
"""
            
            logger.info(f"Calling LLM for extraction (processing {len(markdown_chunk)} chars)...")
            
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-5-nano",
                    messages=[
                        {"role": "system", "content": "You are a data extraction assistant. Extract structured data and return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                extracted = response.choices[0].message.content
                logger.info(f"LLM extraction successful. Result length: {len(extracted)} chars")
                logger.info(f"Extracted data preview: {extracted[:500]}...")
                
                return extracted
                
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
                return "{}"

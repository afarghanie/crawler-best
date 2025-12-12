import os
import asyncio
import logging
from typing import Optional, Dict, Any, Type, List
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy, CosineStrategy
from crawl4ai.async_configs import LLMConfig
import json
import enum

# Configure module logger
logger = logging.getLogger(__name__)

class CrawlingStrategy(str, enum.Enum):
    SIMPLE = "simple"
    DEEP = "deep"
    ADAPTIVE = "adaptive"

class CrawlerEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    def _get_provider_config(self):
        """
        Determines the base_url and model to use based on the API Key.
        Returns (base_url, model)
        """
        if not self.api_key:
             return "https://openrouter.ai/api/v1", "openai/gpt-5-nano"
             
        if self.api_key.startswith("gsk_"):
             return "https://api.groq.com/openai/v1", "llama3-70b-8192"
        elif self.api_key.startswith("sk-proj-") or self.api_key.startswith("sk-"):
             # Simple heuristic for OpenAI keys (sk-...) if not OpenRouter (sk-or-...)
             # Note: OpenRouter keys usually start with sk-or-v1...
             if not self.api_key.startswith("sk-or-"):
                 return "https://api.openai.com/v1", "gpt-4o"

        # Default to OpenRouter
        return "https://openrouter.ai/api/v1", "openai/gpt-5-nano"

    async def run_summary(self, url: str, query: str) -> str:
        """
        Crawls the URL and returns a focused summary based on the user's query.
        """
        logger.info(f"Starting summary crawl for: {url}")
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
            
            logger.info(f"Summary crawl successful. Content length: {len(result.markdown)} chars")
            
            # Use LLM to generate summary
            from openai import OpenAI
            
            logger.info("Calling LLM for smart summarization...")
            if not self.api_key:
                return "Error: API Key missing for summarization."
            
            # Use dynamic provider config
            base_url, model = self._get_provider_config()
            logger.info(f"Using provider: {base_url} with model: {model}")
                
            client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
            
            # Increase limit for summary to capture full article content
            # gpt-5-nano has a large context window, so we can pass more data
            content_limit = 100000 
            markdown_chunk = result.markdown[:content_limit]
            
            logger.info(f"Generating summary with input length: {len(markdown_chunk)} chars")
            
            prompt = f"""Summarize the following website content based on the user's specific request.

User Request: "{query}"

Website Content:
{markdown_chunk}

Instructions:
- Provide a clear, well-formatted markdown summary.
- Focus ONLY on what the user asked for.
- Ignore navigation menus, footers, ads, and irrelevant sidebars.
- If the answer isn't in the content, say so clearly.
"""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant. Summarize web content based on user queries."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                summary = response.choices[0].message.content
                logger.info("Smart summary generated successfully")
                return summary
                
            except Exception as e:
                logger.error(f"LLM summarization failed: {e}")
                return f"Error generating summary: {e}. Returning raw markdown sample:\n\n{result.markdown[:2000]}"

    async def run_extraction(self, url: str, schema: Type[BaseModel], strategy: str = CrawlingStrategy.SIMPLE, max_pages: int = 1, target_count: int = 0) -> str:
        """
        Crawls the URL and extracts data based on the selected strategy.
        
        Args:
            url: The starting URL.
            schema: The Pydantic model for extraction.
            strategy: 'simple', 'deep', or 'adaptive'.
            max_pages: Limit for deep/adaptive crawling.
            target_count: Adaptive strategy stops when this many items are found.
        """
        if not self.api_key:
            logger.error("API Key missing for extraction")
            return "Error: API Key is required for extraction."

        logger.info(f"Starting extraction with strategy: {strategy} for: {url}")

        if strategy == CrawlingStrategy.DEEP:
            return await self._run_deep_extraction(url, schema, max_pages)
        elif strategy == CrawlingStrategy.ADAPTIVE:
            return await self._run_adaptive_extraction(url, schema, max_pages, target_count)
        else:
            return await self._run_simple_extraction(url, schema)

    async def _run_simple_extraction(self, url: str, schema: Type[BaseModel]) -> str:
        """Original single-page extraction logic"""
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=url,
                magic=True,
                cache_mode="BYPASS"
            )
            
            if not result.success:
                return f"Error crawling {url}: {result.error_message}"
            
            return await self._extract_with_llm(result.markdown, schema)

    async def _run_deep_extraction(self, url: str, schema: Type[BaseModel], max_pages: int) -> str:
        """
        Uses Crawl4AI's BestFirstCrawlingStrategy to explore links.
        Merges results from multiple pages.
        """
        from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
        from crawl4ai.async_configs import CrawlerRunConfig
        from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
        
        logger.info(f"Running Deep Crawl (BestFirst) with max_pages={max_pages}")
        
        # Heuristic: Score pages with "page" or "next" or specific schema keywords higher
        schema_keywords = list(schema.model_fields.keys())
        scorer = KeywordRelevanceScorer(keywords=schema_keywords + ["page", "next", "list"], weight=0.8)
        
        strategy = BestFirstCrawlingStrategy(
            max_depth=2,
            include_external=False,
            url_scorer=scorer,
            max_pages=max_pages
        )
        
        config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            cache_mode="BYPASS",
            magic=True
        )
        
        all_items = []
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            async for result in await crawler.arun(url, config=config):
                if result.success:
                   logger.info(f"Deep Crawl visited: {result.url}")
                   extracted_json_str = await self._extract_with_llm(result.markdown, schema)
                   try:
                       data = json.loads(extracted_json_str)
                       if "data" in data and isinstance(data["data"], list):
                           all_items.extend(data["data"])
                   except:
                       logger.warning(f"Failed to parse JSON from {result.url}")
        
        return json.dumps({"data": all_items}, indent=2)

    async def _run_adaptive_extraction(self, start_url: str, schema: Type[BaseModel], max_pages: int, target_count: int) -> str:
        """
        Custom loop: Crawls -> Extracts -> Finds 'Next' -> Repeats
        Stops when max_pages is reached OR target_count is met.
        """
        current_url = start_url
        all_items = []
        visited_urls = set()
        
        pages_crawled = 0
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            while pages_crawled < max_pages:
                if current_url in visited_urls:
                    logger.info(f"Already visited {current_url}, stopping loop.")
                    break
                
                visited_urls.add(current_url)
                pages_crawled += 1
                logger.info(f"Adaptive Crawl ({pages_crawled}/{max_pages}): {current_url}")
                
                result = await crawler.arun(url=current_url, magic=True, cache_mode="BYPASS")
                
                if not result.success:
                    logger.error(f"Failed to crawl {current_url}")
                    break
                
                # 1. Extract Data
                extracted_json_str = await self._extract_with_llm(result.markdown, schema)
                try:
                    data = json.loads(extracted_json_str)
                    items = data.get("data", [])
                    all_items.extend(items)
                    logger.info(f"Found {len(items)} items. Total: {len(all_items)}/{target_count}")
                except Exception as e:
                    logger.warning(f"Failed to parse data: {e}")
                
                # 2. Check Termination
                if target_count > 0 and len(all_items) >= target_count:
                    logger.info("Target count reached!")
                    break
                
                # 3. Find Next Page
                logger.info("Looking for next page link...")
                next_url = await self._find_next_page(result.markdown, current_url)
                if not next_url:
                    logger.info("No next page found. Stopping.")
                    break
                    
                current_url = next_url
                
        return json.dumps({"data": all_items}, indent=2)

    async def _find_next_page(self, markdown: str, current_url: str) -> Optional[str]:
        """Uses LLM to find the 'Next' pagination link from markdown content."""
        from openai import OpenAI
        base_url, model = self._get_provider_config()
        client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        # Optimize context: Take the last 5000 chars where pagination usually is
        footer_content = markdown[-5000:]
        
        prompt = f"""Analyze the following website footer/navigation content and find the URL for the "Next" page.
        
Current URL: {current_url}

Content:
{footer_content}

Instructions:
- Return ONLY the URL for the next page.
- If it's a relative path, make it absolute if possible, or just return the relative path.
- If no "Next" page link is found, return "None".
- Do NOT return markdown formatting, just the raw string.
"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            next_link = response.choices[0].message.content.strip()
            if "None" in next_link or next_link == "":
                return None
            
            # Simple absolute url fix
            if next_link.startswith("/"):
                from urllib.parse import urljoin, urlparse
                parsed_base = urlparse(current_url)
                base = f"{parsed_base.scheme}://{parsed_base.netloc}"
                next_link = urljoin(base, next_link)
                
            return next_link
        except Exception as e:
            logger.error(f"Error finding next page: {e}")
            return None

    async def _extract_with_llm(self, markdown: str, schema: Type[BaseModel]) -> str:
        """Helper to run direct LLM extraction on markdown"""
        from openai import OpenAI
        
        base_url, model = self._get_provider_config()
        client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        schema_json = schema.model_json_schema()
        content_limit = 30000
        markdown_chunk = markdown[:content_limit]
        
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
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Extract structured data and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM extraction helper failed: {e}")
            return "{}"

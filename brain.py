import os
import json
import logging
import re
from typing import Type, List, Optional
from pydantic import BaseModel, create_model
from openai import OpenAI

# Configure module logger
logger = logging.getLogger(__name__)

class StrategyType:
    SUMMARY = "summary"
    EXTRACTION = "extraction"

class IntentAnalyzer:
    def __init__(self, api_key: str):
        # Default to OpenRouter if not Groq/OpenAI specific
        base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/gpt-5-nano"
        
        if api_key.startswith("gsk_"):
             base_url = "https://api.groq.com/openai/v1"
             self.model = "llama3-70b-8192"
        elif api_key.startswith("sk-proj-"): # OpenAI project keys often start with sk-proj
             base_url = "https://api.openai.com/v1"
             self.model = "gpt-4o"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"IntentAnalyzer initialized with model: {self.model}")

    def analyze(self, prompt: str) -> str:
        """
        Determines if the user wants a summary or specific data extraction.
        """
        logger.info(f"Analyzing intent for prompt: '{prompt}'")
        system_prompt = """
        You are an intent classifier for a web scraping agent.
        Analyze the user's prompt and categorize it as either 'summary' or 'extraction'.
        
        - 'summary': The user wants a general overview, summary, or full text of the page.
        - 'extraction': The user wants specific fields, data points, or a list of items.
        
        Respond ONLY with the JSON: {"strategy": "summary"} or {"strategy": "extraction"}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            strategy = result.get("strategy", StrategyType.SUMMARY)
            logger.info(f"Intent identified: {strategy}")
            return strategy
        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            return StrategyType.SUMMARY

class SchemaGenerator:
    def __init__(self, api_key: str):

        # Default to OpenRouter if not Groq/OpenAI specific
        base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/gpt-5-nano"
        
        if api_key.startswith("gsk_"):
             base_url = "https://api.groq.com/openai/v1"
             self.model = "llama3-70b-8192"
        elif api_key.startswith("sk-proj-"):
             base_url = "https://api.openai.com/v1"
             self.model = "gpt-4o"
             
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"SchemaGenerator initialized with model: {self.model}")

    def generate_schema(self, prompt: str) -> Type[BaseModel]:
        """
        Generates a Pydantic model based on the user's extraction request.
        """
        logger.info(f"Generating schema for prompt: '{prompt}'")
        system_prompt = """
        You are a data schema expert. Convert the user's data extraction request into a JSON schema definition.
        The output must be a valid JSON object describing the fields, types, and structure.
        
        Example User Request: "Get the product name, price, and rating."
        Example JSON Output:
        {
            "product_name": "string",
            "price": "string",
            "rating": "number"
        }
        
        If the user asks for a list of items, define the schema for a SINGLE item. The system will handle the list wrapping.
        Keep field names snake_case. Use simple types: string, number, boolean, array.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # Use regex to find the first JSON object or array
            match = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
            if match:
                content = match.group(0)
            else:
                 # If no JSON found, try to clean markdown
                content = content.replace("```json", "").replace("```", "").strip()
            
            schema_json = json.loads(content)
            logger.info(f"Generated schema JSON: {json.dumps(schema_json)}")
            
            # Dynamically create Pydantic model
            fields = {}
            for name, type_str in schema_json.items():
                if type_str == "number":
                    fields[name] = (float, ...)
                elif type_str == "boolean":
                    fields[name] = (bool, ...)
                elif type_str == "array":
                     fields[name] = (List[str], ...) # Defaulting to list of strings for simplicity
                else:
                    fields[name] = (str, ...)
            
            return create_model('DynamicSchema', **fields)
            
        except Exception as e:
            logger.error(f"Error in schema generation: {e}")
            # Fallback schema
            return create_model('FallbackSchema', content=(str, ...))

class ConfigExtractor:
    def __init__(self, api_key: str):
        # Default to OpenRouter if not Groq/OpenAI specific
        base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/gpt-5-nano"
        
        if api_key.startswith("gsk_"):
             base_url = "https://api.groq.com/openai/v1"
             self.model = "llama3-70b-8192"
        elif api_key.startswith("sk-proj-"):
             base_url = "https://api.openai.com/v1"
             self.model = "gpt-4o"
             
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"ConfigExtractor initialized with model: {self.model}")

    def extract_config(self, prompt: str) -> dict:
        """
        Extracts execution configuration (strategy, limits) from the user's prompt.
        """
        logger.info(f"Extracting config for prompt: '{prompt}'")
        system_prompt = """
        You are a configuration expert for a web crawler. Analyze the user's request and determine the best execution strategy.
        
        Strategies:
        - "simple": Default. Single page, specific data.
        - "deep": User asks to "explore", "scan site", "crawl deep", "find all pages".
        - "adaptive": User specifies a quantity (e.g., "get 50 items", "100 quotes") that might span multiple pages.
        
        Output JSON:
        {
            "strategy": "simple" | "deep" | "adaptive",
            "target_count": number (0 if not specified),
            "max_pages": number (1 for simple, 5-10 for others depending on inferred need)
        }
        
        Rules:
        1. If user asks for a specific count (e.g. "30 items"), set "strategy": "adaptive", "target_count": 30, "max_pages": 10.
        2. If user asks for "all" items, set "strategy": "deep", "max_pages": 5.
        3. If vague, default to "simple".
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            config = json.loads(response.choices[0].message.content)
            
            # Sanity checks
            if config.get("strategy") not in ["simple", "deep", "adaptive"]:
                config["strategy"] = "simple"
                
            logger.info(f"Extracted config: {config}")
            return config
            
        except Exception as e:
            logger.error(f"Error extracting config: {e}")
            return {"strategy": "simple", "target_count": 0, "max_pages": 1}

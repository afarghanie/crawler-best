import streamlit as st
import asyncio
import nest_asyncio
import pandas as pd
import json
import logging
import io
from brain import IntentAnalyzer, SchemaGenerator, StrategyType, ConfigExtractor
from d_crawler import CrawlerEngine

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Patch asyncio for Streamlit
nest_asyncio.apply()

# Configure Logging
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
        # Update container
        self.container.code("\n".join(self.logs), language="text")

st.set_page_config(page_title="IntelliCrawl", page_icon="🕷️", layout="wide")

st.title("🕷️ IntelliCrawl: The Text-to-Data Agent")
st.markdown("Extract data from any website using natural language.")

# Load API key from environment (no UI display)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing `OPENROUTER_API_KEY` in .env file.")
    st.stop()

# Setup Logging (log_container will be created later in the flow)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Clear existing handlers to avoid duplicates on rerun
if logger.hasHandlers():
    logger.handlers.clear()


# Main Interface
st.text_input("Target URL", placeholder="https://example.com", key="url_input")
st.text_area("What do you want to extract?", placeholder="e.g., Get me 20 list of second hand car...", key="prompt_input")
run_btn = st.button("Run Crawler", type="primary")

# Access values safely
url = st.session_state.get("url_input", "")
prompt = st.session_state.get("prompt_input", "")

if run_btn and url and prompt:
    # --- LOGGING SETUP (Initialize immediately) ---
    st.subheader("Session Logs")
    log_container = st.empty()
    
    # Setup logging handler
    handler = StreamlitLogHandler(log_container)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # ----------------------------------------------

    try:
        # 1. Intent Analysis
        logger.info("🧠 Analyzing Intent...")
        logger.info(f"User initiated crawl. URL: {url}")
        
        analyzer = IntentAnalyzer(api_key)
        strategy = analyzer.analyze(prompt)
        st.write(f"**Strategy Detected:** `{strategy}`")
        
        crawler = CrawlerEngine(api_key)
        
        if strategy == StrategyType.SUMMARY:
            # 2. Summary Execution
            logger.info("🕷️ Crawling for Summary...")
            result = asyncio.run(crawler.run_summary(url, prompt))
            
            st.subheader("Result")
            st.markdown(result)
            
            st.download_button("Download Markdown", result, file_name="summary.md")
            
        else:
            # 2. Extraction Execution
            logger.info("🧠 Generating Schema & Config...")
            generator = SchemaGenerator(api_key)
            schema = generator.generate_schema(prompt)
            
            # --- Extract Configuration ---
            config_extractor = ConfigExtractor(api_key)
            config = config_extractor.extract_config(prompt)
            
            crawl_strategy = config.get("strategy", "simple")
            target_count = config.get("target_count", 0)
            max_pages = config.get("max_pages", 1)
            
            st.write(f"**Execution Plan:** Strategy=`{crawl_strategy}`, Target=`{target_count}`, Max Pages=`{max_pages}`")
            # -----------------------------
            
            # Show Generated Schema
            with st.expander("View Generated Schema"):
                st.json(schema.model_json_schema())
            
            logger.info(f"🕷️ Crawling... ({crawl_strategy})")
            json_result = asyncio.run(crawler.run_extraction(
                url=url, 
                schema=schema,
                strategy=crawl_strategy,
                max_pages=max_pages,
                target_count=target_count
            ))
            
            st.subheader("Extracted Data")
            try:
                if not json_result:
                    raise ValueError("Empty result from crawler")
                    
                data = json.loads(json_result)
                
                # Handle the new format with "data" key
                if isinstance(data, dict) and "data" in data:
                    items = data["data"]
                elif isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                if len(items) > 0:
                    df = pd.DataFrame(items)
                    st.dataframe(df, use_container_width=True)
                    logger.info(f"✅ Extracted {len(items)} items successfully!")
                    
                    # Download buttons
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", csv, "data.csv", "text/csv")
                    with col_d2:
                        json_str = json.dumps(items, indent=2)
                        st.download_button("Download JSON", json_str, "data.json", "application/json")
                else:
                    logger.warning("No data extracted")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error: {e}")
                st.text(json_result)
                
        logger.info("Process finished successfully.")
            
    except Exception as e:
        logger.error(f"Critical Error: {e}")
        st.error(f"An error occurred: {e}")

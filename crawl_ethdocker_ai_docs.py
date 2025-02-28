import os
import sys
import json
import asyncio
import requests
import hashlib
import tenacity
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
import re
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    content_hash: str  # For version tracking
    parent_chunk: Optional[str]  # Reference to previous chunk for context
    next_chunk: Optional[str]  # Reference to next chunk for context
    section_hierarchy: List[str]  # Document section hierarchy
    keywords: List[str]  # Extracted keywords
    last_updated: str  # Timestamp of last update
    version: int  # Document version number

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better embedding quality."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('—', '-')
    # Remove URLs while preserving important parts
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                  lambda m: urlparse(m.group()).path, text)
    return text.strip()

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text using basic NLP techniques."""
    # Remove common code symbols and split
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned.split()
    # Remove common stop words (you might want to use a proper NLP library for this)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    # Return unique keywords
    return list(set(keywords))

def get_content_hash(content: str) -> str:
    """Generate a hash of the content for version tracking."""
    return hashlib.sha256(content.encode()).hexdigest()

def extract_section_hierarchy(text: str) -> List[str]:
    """Extract section hierarchy from markdown headings."""
    hierarchy = []
    for line in text.split('\n'):
        if line.strip().startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line.strip('#').strip()
            while len(hierarchy) >= level:
                hierarchy.pop()
            hierarchy.append(title)
    return hierarchy

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks while preserving semantic coherence."""
    chunks = []
    start = 0
    text_length = len(text)
    current_section = []

    while start < text_length:
        # Calculate end position with overlap
        end = start + chunk_size
        
        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunk_text = text[start:].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start": start,
                    "end": text_length,
                    "section_hierarchy": extract_section_hierarchy(chunk_text)
                })
            break

        # Look for semantic boundaries in order of preference
        boundaries = {
            'header': text[start:end].rfind('\n#'),
            'code_block': text[start:end].rfind('\n```'),
            'paragraph': text[start:end].rfind('\n\n'),
            'sentence': text[start:end].rfind('. ')
        }

        # Choose the best boundary that's not too close to the start
        chosen_end = end
        min_chunk_size = chunk_size * 0.5  # Minimum 50% of target size
        
        for boundary_type, boundary_pos in boundaries.items():
            if boundary_pos > start + min_chunk_size:
                chosen_end = start + boundary_pos
                break

        # Extract chunk with proper context
        chunk_text = text[start:chosen_end].strip()
        if chunk_text:
            # Get section hierarchy for this chunk
            section_hierarchy = extract_section_hierarchy(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": chosen_end,
                "section_hierarchy": section_hierarchy
            })

        # Move start position for next chunk, including overlap
        start = max(start + 1, chosen_end - overlap)

    # Link chunks together
    for i in range(len(chunks)):
        if i > 0:
            chunks[i]["prev_chunk_id"] = i - 1
        if i < len(chunks) - 1:
            chunks[i]["next_chunk_id"] = i + 1

    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((requests.RequestException, asyncio.TimeoutError))
)
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4 with retry mechanism."""
    try:
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""
        
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={ "type": "json_object" },
            timeout=30
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error getting title and summary for {url}: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((requests.RequestException, asyncio.TimeoutError))
)
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI with retry mechanism."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            timeout=30
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

async def process_chunk(chunk_data: Dict[str, Any], url: str, chunk_number: int) -> ProcessedChunk:
    """Process a single chunk of text with enhanced context."""
    # Preprocess the text
    processed_text = preprocess_text(chunk_data["text"])
    
    # Get title and summary
    extracted = await get_title_and_summary(processed_text, url)
    
    # Extract keywords
    keywords = extract_keywords(processed_text)
    
    # Get embedding for processed text
    embedding = await get_embedding(processed_text)
    
    # Create metadata with enhanced context
    metadata = {
        "source": "ethdocker_docs",
        "chunk_size": len(processed_text),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
        "start_pos": chunk_data["start"],
        "end_pos": chunk_data["end"],
        "semantic_boundaries": {
            "prev_chunk_id": chunk_data.get("prev_chunk_id"),
            "next_chunk_id": chunk_data.get("next_chunk_id")
        }
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=processed_text,
        metadata=metadata,
        embedding=embedding,
        content_hash=get_content_hash(processed_text),
        parent_chunk=str(chunk_data.get("prev_chunk_id")),
        next_chunk=str(chunk_data.get("next_chunk_id")),
        section_hierarchy=chunk_data["section_hierarchy"],
        keywords=keywords,
        last_updated=datetime.now(timezone.utc).isoformat(),
        version=1
    )

async def insert_chunk_with_conflict_resolution(chunk: ProcessedChunk) -> Optional[Dict]:
    """Insert a chunk with conflict resolution for existing content."""
    try:
        # Check if a chunk with the same URL and number exists
        existing = supabase.table("ethdocker_site_pages").select("*").eq(
            "url", chunk.url
        ).eq("chunk_number", chunk.chunk_number).execute()

        if existing.data:
            existing_chunk = existing.data[0]
            if existing_chunk["content_hash"] != chunk.content_hash:
                # Content has changed, update with new version
                chunk.version = existing_chunk["version"] + 1
                data = chunk.__dict__
                result = supabase.table("ethdocker_site_pages").update(
                    data
                ).eq("url", chunk.url).eq(
                    "chunk_number", chunk.chunk_number
                ).execute()
                logger.info(f"Updated chunk {chunk.chunk_number} for {chunk.url} (version {chunk.version})")
            else:
                logger.info(f"Chunk {chunk.chunk_number} for {chunk.url} unchanged")
                return existing.data[0]
        else:
            # New chunk, insert it
            data = chunk.__dict__
            result = supabase.table("ethdocker_site_pages").insert(data).execute()
            logger.info(f"Inserted new chunk {chunk.chunk_number} for {chunk.url}")
            return result.data[0]
            
    except Exception as e:
        logger.error(f"Error inserting/updating chunk: {str(e)}")
        raise

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks with error handling."""
    try:
        # Split into chunks
        chunks = chunk_text(markdown)
        logger.info(f"Processing {len(chunks)} chunks for {url}")
        
        # Process chunks in parallel with semaphore to limit concurrency
        sem = asyncio.Semaphore(5)  # Limit concurrent API calls
        
        async def process_chunk_with_semaphore(chunk_data: Dict, i: int) -> Tuple[int, Optional[ProcessedChunk]]:
            async with sem:
                try:
                    return i, await process_chunk(chunk_data, url, i)
                except Exception as e:
                    logger.error(f"Error processing chunk {i} for {url}: {str(e)}")
                    return i, None

        # Process chunks with controlled concurrency
        chunk_tasks = [
            process_chunk_with_semaphore(chunk, i) 
            for i, chunk in enumerate(chunks)
        ]
        processed_results = await asyncio.gather(*chunk_tasks)
        
        # Filter out failed chunks and sort by index
        processed_chunks = [
            chunk for _, chunk in sorted(processed_results, key=lambda x: x[0])
            if chunk is not None
        ]
        
        # Store chunks with conflict resolution
        store_tasks = [
            insert_chunk_with_conflict_resolution(chunk) 
            for chunk in processed_chunks
        ]
        await asyncio.gather(*store_tasks)
        
        logger.info(f"Successfully processed and stored {len(processed_chunks)} chunks for {url}")
        
    except Exception as e:
        logger.error(f"Error processing document {url}: {str(e)}")
        raise

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_ethdocker_docs_urls() -> List[str]:
    """Get URLs from ethdocker AI docs sitemap."""
    sitemap_url = "https://ethdocker.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs from ethdocker AI docs
    urls = get_ethdocker_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
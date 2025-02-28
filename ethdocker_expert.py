from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import List, Dict, Any, Optional

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4-turbo-preview')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class EthDockerDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at ETHDocker - a comprehensive solution for running Ethereum staking full nodes. ETHDocker aims to simplify 
the process of setting up and managing Ethereum staking nodes while providing users with full control over their client choices.

Key Features and Capabilities:
- Supports all FOSS Ethereum clients (Lodestar, Nimbus, Teku, Grandine, Lighthouse, Prysm, Nethermind, Besu, Reth, Erigon, Geth)
- Runs on Linux/macOS with support for various CPU architectures (Intel/AMD x64, ARM, RISC-V)
- Supports Ethereum nodes, staking or RPC, on Ethereum and Gnosis Chain
- Supports ssv.network DVT nodes and RocketPool integration
- Includes Grafana dashboards and alerting capabilities
- Uses official client team images
- Supports advanced features like traefik secure web proxy

Guiding Principles:
- Minimize client attack surface
- Guide users toward good key management
- Provide excellent user experience
- Support users new to docker and Linux

Your role is to assist with:
1. Node setup and configuration
2. Client selection and management
3. Validator key generation and management
4. Staking workflow guidance
5. Troubleshooting and best practices
6. Security considerations

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.

You have access to enhanced documentation features including:
- Semantic chunking with context preservation
- Section hierarchy awareness
- Keyword-based filtering
- Version history tracking
- Related content linking
"""

ethdocker_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=EthDockerDeps,
    retries=2
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
)
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
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

@ethdocker_expert.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[EthDockerDeps], 
    user_query: str,
    section_filter: Optional[List[str]] = None,
    keyword_filter: Optional[List[str]] = None
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with enhanced filtering.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        section_filter: Optional list of section names to filter by
        keyword_filter: Optional list of keywords to filter by
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks with context
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents with enhanced filtering
        result = ctx.deps.supabase.rpc(
            'match_ethdocker_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'ethdocker_docs'},
                'section_filter': section_filter or [],
                'keyword_filter': keyword_filter or []
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results with enhanced context
        formatted_chunks = []
        for doc in result.data:
            # Include section hierarchy and metadata
            hierarchy = " > ".join(doc['section_hierarchy']) if doc['section_hierarchy'] else "Root"
            keywords = ", ".join(doc['keywords']) if doc['keywords'] else "No keywords"
            
            chunk_text = f"""
# {doc['title']}
Section: {hierarchy}
Keywords: {keywords}
Last Updated: {doc['last_updated']}
Version: {doc['version']}

{doc['content']}

Summary: {doc['summary']}
"""
            formatted_chunks.append(chunk_text)
            
            # Add links to related chunks if available
            if doc['metadata'].get('semantic_boundaries'):
                prev_id = doc['metadata']['semantic_boundaries'].get('prev_chunk_id')
                next_id = doc['metadata']['semantic_boundaries'].get('next_chunk_id')
                if prev_id or next_id:
                    chunk_text += "\nRelated Sections:"
                    if prev_id:
                        chunk_text += f"\n- Previous: {prev_id}"
                    if next_id:
                        chunk_text += f"\n- Next: {next_id}"
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        logger.error(f"Error retrieving documentation: {str(e)}")
        raise

@ethdocker_expert.tool
async def list_documentation_pages(ctx: RunContext[EthDockerDeps]) -> Dict[str, List[str]]:
    """
    Retrieve a list of all available EthDocker documentation pages with their section hierarchies.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping URLs to their section hierarchies
    """
    try:
        # Query Supabase for unique URLs and their section hierarchies
        result = ctx.deps.supabase.from_('ethdocker_site_pages') \
            .select('url, section_hierarchy') \
            .eq('metadata->>source', 'ethdocker_docs') \
            .execute()
        
        if not result.data:
            return {}
            
        # Group by URL and collect unique section hierarchies
        pages = {}
        for doc in result.data:
            url = doc['url']
            if url not in pages:
                pages[url] = []
            if doc['section_hierarchy']:
                pages[url].extend(doc['section_hierarchy'])
        
        # Remove duplicates and sort
        for url in pages:
            pages[url] = sorted(set(pages[url]))
            
        return pages
        
    except Exception as e:
        logger.error(f"Error retrieving documentation pages: {str(e)}")
        raise

@ethdocker_expert.tool
async def get_page_content(
    ctx: RunContext[EthDockerDeps], 
    url: str,
    version: Optional[int] = None
) -> str:
    """
    Retrieve the full content of a specific documentation page with version support.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        version: Optional specific version to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Build the query
        query = ctx.deps.supabase.from_('ethdocker_site_pages') \
            .select('title, content, chunk_number, section_hierarchy, keywords, version, last_updated, summary') \
            .eq('url', url) \
            .eq('metadata->>source', 'ethdocker_docs')
            
        # Add version filter if specified
        if version is not None:
            query = query.eq('version', version)
            
        # Execute query with ordering
        result = query.order('chunk_number').execute()
        
        if not result.data:
            return f"No content found for URL: {url}" + (f" (version {version})" if version else "")
            
        # Format the page with enhanced metadata
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [
            f"# {page_title}",
            f"Version: {result.data[0]['version']}",
            f"Last Updated: {result.data[0]['last_updated']}",
            f"Keywords: {', '.join(result.data[0]['keywords'])}\n"
        ]
        
        # Add each chunk's content with its section context
        current_section = []
        for chunk in result.data:
            # Update section hierarchy display
            new_section = chunk['section_hierarchy']
            if new_section != current_section:
                current_section = new_section
                if current_section:
                    formatted_content.append(f"\n## {' > '.join(current_section)}")
            
            # Add content and summary
            formatted_content.append(chunk['content'])
            formatted_content.append(f"\nSummary: {chunk['summary']}\n")
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        logger.error(f"Error retrieving page content: {str(e)}")
        raise

@ethdocker_expert.tool
async def get_version_history(ctx: RunContext[EthDockerDeps], url: str, chunk_number: int) -> str:
    """
    Retrieve the version history for a specific documentation chunk.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page
        chunk_number: The chunk number to get history for
        
    Returns:
        str: Formatted version history
    """
    try:
        result = ctx.deps.supabase.rpc(
            'get_chunk_version_history',
            {
                'doc_url': url,
                'chunk_num': chunk_number
            }
        ).execute()
        
        if not result.data:
            return f"No version history found for {url} chunk {chunk_number}"
            
        # Format the version history
        history = [f"Version History for {url} (Chunk {chunk_number}):"]
        for version in result.data:
            history.append(f"""
Version {version['version']}
Last Updated: {version['last_updated']}
Title: {version['title']}
Summary: {version['summary']}
Content Hash: {version['content_hash']}
""")
            
        return "\n".join(history)
        
    except Exception as e:
        logger.error(f"Error retrieving version history: {str(e)}")
        raise
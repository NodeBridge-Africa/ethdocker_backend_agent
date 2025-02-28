# ETHDocker Documentation Crawler and Knowledge Base

A powerful documentation crawler and knowledge base system that processes and stores documentation from ethdocker.com with advanced semantic search capabilities.

## Features

- 🕷️ Asynchronous web crawling with parallel processing
- 🧠 Semantic chunking with context preservation
- 🔍 Advanced vector search using OpenAI embeddings
- 📚 Version control and document history tracking
- 🔗 Hierarchical document structure with linked chunks
- 🏷️ Automatic keyword extraction and categorization
- ⚡ High-performance PostgreSQL storage with pgvector
- 🔄 Intelligent conflict resolution and version management

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Supabase account (for hosted database)
- OpenAI API key

## Installation

1. Clone the repository and set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

3. Configure your `.env` file with:

```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
LLM_MODEL=gpt-4-turbo-preview  # or your preferred model
```

4. Set up the database schema:

```bash
# Using psql or your preferred PostgreSQL client
psql -d your_database -f site_pages.sql
```

## Usage

Run the crawler to fetch and process documentation:

```bash
python crawl_ethdocker_ai_docs.py
```

The crawler will:

1. Fetch URLs from the ethdocker.com sitemap
2. Process documents in parallel with controlled concurrency
3. Split content into semantic chunks with context preservation
4. Generate embeddings and extract metadata
5. Store processed content with version control

## Database Schema

The system uses a PostgreSQL database with the following key features:

- Vector similarity search using pgvector
- Full-text search capabilities
- Document version history
- Hierarchical document structure
- Keyword-based filtering
- Metadata-based querying

### Key Functions

- `match_ethdocker_site_pages`: Search documents by semantic similarity
- `get_chunk_version_history`: Retrieve version history for document chunks

## Architecture

### Document Processing

1. **Crawling**: Asynchronous crawling with rate limiting and error handling
2. **Chunking**: Smart text splitting with semantic boundary detection
3. **Enrichment**:
   - Title and summary generation using GPT-4
   - Keyword extraction
   - Section hierarchy tracking
   - Embedding generation
4. **Storage**:
   - Conflict resolution
   - Version management
   - Linked chunk references

### Performance Optimizations

- Parallel processing with controlled concurrency
- Efficient database indexing
- Caching and retry mechanisms
- Batch operations for better throughput

## Error Handling

The system includes:

- Automatic retries with exponential backoff
- Comprehensive logging
- Transaction management
- Conflict resolution
- Failure recovery

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenAI for embedding and GPT-4 APIs
- Supabase for hosted PostgreSQL
- pgvector for vector similarity search

-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table ethdocker_site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    content_hash varchar(64) not null,  -- SHA-256 hash
    parent_chunk varchar,  -- Reference to previous chunk
    next_chunk varchar,  -- Reference to next chunk
    section_hierarchy text[] not null default array[]::text[],  -- Document section hierarchy
    keywords text[] not null default array[]::text[],  -- Extracted keywords
    last_updated timestamp with time zone not null,
    version integer not null default 1,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add constraints
    unique(url, chunk_number),
    check (version > 0)
);

-- Create indexes for better query performance
create index on ethdocker_site_pages using ivfflat (embedding vector_cosine_ops);
create index idx_ethdocker_site_pages_metadata on ethdocker_site_pages using gin (metadata);
create index idx_ethdocker_site_pages_content_hash on ethdocker_site_pages(content_hash);
create index idx_ethdocker_site_pages_section_hierarchy on ethdocker_site_pages using gin (section_hierarchy);
create index idx_ethdocker_site_pages_keywords on ethdocker_site_pages using gin (keywords);
create index idx_ethdocker_site_pages_url_version on ethdocker_site_pages(url, version);

-- Create a function to search for documentation chunks with enhanced filtering
create or replace function match_ethdocker_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  section_filter text[] DEFAULT array[]::text[],
  keyword_filter text[] DEFAULT array[]::text[]
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  section_hierarchy text[],
  keywords text[],
  version integer,
  last_updated timestamp with time zone,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    section_hierarchy,
    keywords,
    version,
    last_updated,
    1 - (ethdocker_site_pages.embedding <=> query_embedding) as similarity
  from ethdocker_site_pages
  where metadata @> filter
    and (array_length(section_filter, 1) is null 
         or section_hierarchy && section_filter)
    and (array_length(keyword_filter, 1) is null 
         or keywords && keyword_filter)
  order by ethdocker_site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create a function to get document version history
create or replace function get_chunk_version_history(
  doc_url varchar,
  chunk_num integer
) returns table (
  version integer,
  content_hash varchar,
  last_updated timestamp with time zone,
  title varchar,
  summary varchar
)
language sql
as $$
  select 
    version,
    content_hash,
    last_updated,
    title,
    summary
  from ethdocker_site_pages
  where url = doc_url and chunk_number = chunk_num
  order by version desc;
$$;

-- Enable RLS on the table
alter table ethdocker_site_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on ethdocker_site_pages
  for select
  to public
  using (true);
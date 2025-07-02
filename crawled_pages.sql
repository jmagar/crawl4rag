-- Enable the pgvector extension
create extension if not exists vector;

-- Drop tables if they exist (to allow rerunning the script)
drop table if exists crawled_pages;
drop table if exists code_examples;
drop table if exists sources;

-- Create application user role for MCP server
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mcp_app_user') THEN
        CREATE ROLE mcp_app_user WITH LOGIN ENCRYPTED PASSWORD 'mcp_secure_password_change_me';
    END IF;
END
$$;

-- Create the sources table
create table sources (
    source_id text primary key,
    summary text,
    total_word_count integer default 0,
    owner_id text not null default current_user,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create the documentation chunks table
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    owner_id text not null default current_user,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    foreign key (source_id) references sources(source_id)
);

-- Create an index for better vector similarity search performance
create index on crawled_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages (source_id);

-- Create composite indexes for common query patterns
CREATE INDEX idx_crawled_pages_source_content ON crawled_pages (source_id, owner_id);
CREATE INDEX idx_crawled_pages_url_chunk ON crawled_pages (url, chunk_number);
CREATE INDEX idx_crawled_pages_owner ON crawled_pages (owner_id);

-- Create a function to search for documentation chunks with proper authorization
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
security definer
as $$
#variable_conflict use_column
begin
  return query
  select
    crawled_pages.id,
    crawled_pages.url,
    crawled_pages.chunk_number,
    crawled_pages.content,
    crawled_pages.metadata,
    crawled_pages.source_id,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where crawled_pages.metadata @> filter
    AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
    AND (crawled_pages.owner_id = current_user OR current_user = 'mcp_app_user')
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the crawled_pages table
alter table crawled_pages enable row level security;

-- Create policies for crawled_pages with proper authentication
create policy "Users can access their own data"
  on crawled_pages
  for all
  to public
  using (owner_id = current_user);

create policy "MCP app can access all data"
  on crawled_pages
  for all
  to mcp_app_user
  using (true);

-- Enable RLS on the sources table
alter table sources enable row level security;

-- Create policies for sources with proper authentication
create policy "Users can access their own sources"
  on sources
  for all
  to public
  using (owner_id = current_user);

create policy "MCP app can access all sources"
  on sources
  for all
  to mcp_app_user
  using (true);

-- Create the code_examples table
create table code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- The code example content
    summary text not null,  -- Summary of the code example
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    owner_id text not null default current_user,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    foreign key (source_id) references sources(source_id)
);

-- Create an index for better vector similarity search performance
create index on code_examples using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_code_examples_metadata on code_examples using gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_code_examples_source_id ON code_examples (source_id);

-- Create composite indexes for code examples
CREATE INDEX idx_code_examples_source_owner ON code_examples (source_id, owner_id);
CREATE INDEX idx_code_examples_owner ON code_examples (owner_id);

-- Create a function to search for code examples with proper authorization
create or replace function match_code_examples (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
security definer
as $$
#variable_conflict use_column
begin
  return query
  select
    code_examples.id,
    code_examples.url,
    code_examples.chunk_number,
    code_examples.content,
    code_examples.summary,
    code_examples.metadata,
    code_examples.source_id,
    1 - (code_examples.embedding <=> query_embedding) as similarity
  from code_examples
  where code_examples.metadata @> filter
    AND (source_filter IS NULL OR code_examples.source_id = source_filter)
    AND (code_examples.owner_id = current_user OR current_user = 'mcp_app_user')
  order by code_examples.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the code_examples table
alter table code_examples enable row level security;

-- Create policies for code_examples with proper authentication
create policy "Users can access their own code examples"
  on code_examples
  for all
  to public
  using (owner_id = current_user);

create policy "MCP app can access all code examples"
  on code_examples
  for all
  to mcp_app_user
  using (true);

-- Grant permissions to MCP app user
GRANT USAGE ON SCHEMA public TO mcp_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON sources TO mcp_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON crawled_pages TO mcp_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON code_examples TO mcp_app_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO mcp_app_user;
GRANT EXECUTE ON FUNCTION match_crawled_pages TO mcp_app_user;
GRANT EXECUTE ON FUNCTION match_code_examples TO mcp_app_user;
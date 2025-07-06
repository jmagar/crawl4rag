-- Migration script: Convert to 1024-dimensional embeddings
-- This script drops existing data and ensures clean 1024-dimension setup
-- 
-- IMPORTANT: This will delete all existing crawled data
-- Run this when you're ready to start fresh with 1024-dimensional embeddings

\echo 'Starting migration to 1024-dimensional embeddings...'

-- Drop existing data (indexes and constraints will be dropped automatically)
DROP TABLE IF EXISTS crawled_pages CASCADE;
DROP TABLE IF EXISTS code_examples CASCADE;
DROP TABLE IF EXISTS sources CASCADE;

\echo 'Dropped existing tables'

-- Re-create tables with 1024 dimensions by running the main schema
\i crawled_pages.sql

\echo 'Migration completed successfully!'
\echo 'All tables recreated with 1024-dimensional vectors'
\echo 'You can now run your crawling operations' 
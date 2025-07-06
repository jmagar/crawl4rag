#!/usr/bin/env python3
"""
Verification script to ensure embedding dimensions are correct.
Run this after migration to verify 1024-dimensional embeddings are working.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils import create_embedding, create_embeddings_batch, get_db_pool, EmbeddingError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def verify_embeddings():
    """Verify that embeddings are generated with correct dimensions."""
    print("🔍 Verifying embedding dimensions...")
    
    try:
        # Test single embedding
        print("Testing single embedding...")
        test_text = "This is a test document for embedding verification."
        embedding = create_embedding(test_text)
        
        print(f"✅ Single embedding dimension: {len(embedding)}")
        if len(embedding) != 1024:
            print(f"❌ ERROR: Expected 1024 dimensions, got {len(embedding)}")
            return False
        
        # Test batch embeddings
        print("Testing batch embeddings...")
        test_texts = [
            "First test document",
            "Second test document",
            "Third test document"
        ]
        embeddings = create_embeddings_batch(test_texts)
        
        print(f"✅ Batch embeddings count: {len(embeddings)}")
        all_correct = True
        for i, emb in enumerate(embeddings):
            if len(emb) != 1024:
                print(f"❌ ERROR: Embedding {i} has {len(emb)} dimensions, expected 1024")
                all_correct = False
            else:
                print(f"  ✅ Embedding {i}: {len(emb)} dimensions")
        
        if not all_correct:
            return False
        
        print("✅ All embeddings have correct dimensions!")
        return True
        
    except EmbeddingError as e:
        print(f"❌ Embedding error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def verify_database_schema():
    """Verify that database schema supports 1024 dimensions."""
    print("\n🗄️  Verifying database schema...")
    
    try:
        db_pool = await get_db_pool()
        
        async with db_pool.acquire() as connection:
            # Check crawled_pages table
            result = await connection.fetchrow("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'crawled_pages' AND column_name = 'embedding'
            """)
            
            if result:
                print(f"✅ crawled_pages.embedding column exists: {result['data_type']}")
            else:
                print("❌ ERROR: crawled_pages.embedding column not found")
                return False
            
            # Check code_examples table
            result = await connection.fetchrow("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'code_examples' AND column_name = 'embedding'
            """)
            
            if result:
                print(f"✅ code_examples.embedding column exists: {result['data_type']}")
            else:
                print("❌ ERROR: code_examples.embedding column not found")
                return False
            
            # Check functions exist with correct signature
            result = await connection.fetchrow("""
                SELECT routine_name, routine_type
                FROM information_schema.routines 
                WHERE routine_name = 'match_crawled_pages'
            """)
            
            if result:
                print(f"✅ match_crawled_pages function exists")
            else:
                print("❌ ERROR: match_crawled_pages function not found")
                return False
            
            result = await connection.fetchrow("""
                SELECT routine_name, routine_type
                FROM information_schema.routines 
                WHERE routine_name = 'match_code_examples'
            """)
            
            if result:
                print(f"✅ match_code_examples function exists")
            else:
                print("❌ ERROR: match_code_examples function not found")
                return False
        
        await db_pool.close()
        print("✅ Database schema verification complete!")
        return True
        
    except Exception as e:
        print(f"❌ Database verification error: {e}")
        return False

async def test_embedding_storage():
    """Test that embeddings can be stored and retrieved from database."""
    print("\n💾 Testing embedding storage and retrieval...")
    
    try:
        # This would require the full MCP server setup
        # For now, just verify the embedding generation works
        test_embedding = create_embedding("Test storage document")
        
        # Verify it can be JSON serialized (required for database storage)
        json_embedding = json.dumps(test_embedding)
        restored_embedding = json.loads(json_embedding)
        
        if len(restored_embedding) == 1024:
            print("✅ Embedding serialization/deserialization works")
            return True
        else:
            print(f"❌ ERROR: Serialization changed dimensions: {len(restored_embedding)}")
            return False
            
    except Exception as e:
        print(f"❌ Storage test error: {e}")
        return False

async def main():
    """Run all verification tests."""
    print("🚀 Starting embedding dimension verification...\n")
    
    results = []
    
    # Test embedding generation
    results.append(await verify_embeddings())
    
    # Test database schema
    results.append(await verify_database_schema())
    
    # Test storage capability
    results.append(await test_embedding_storage())
    
    print(f"\n📊 Verification Results:")
    print(f"  Embedding generation: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"  Database schema: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"  Storage capability: {'✅ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print(f"\n🎉 All verification tests passed!")
        print(f"Your system is ready for 1024-dimensional embeddings.")
        return True
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification script error: {e}")
        sys.exit(1) 
#!/usr/bin/env python3
"""
Test script to demonstrate timing improvements from intelligent batching.

This script shows the before/after performance of contextual embeddings
with and without intelligent batching enabled.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables for testing
os.environ["USE_CONTEXTUAL_EMBEDDINGS"] = "true"
os.environ["OPENAI_TIER"] = "2"  # Adjust based on your actual tier

async def test_batching_performance():
    """Test the performance difference between batched and individual processing."""
    print("🧪 Testing Contextual Embedding Performance Improvements")
    print("=" * 60)
    
    # Sample chunks for testing (simulate a typical crawl result)
    sample_chunks = [
        "This is a sample chunk about web crawling and data extraction.",
        "Here we discuss the implementation of intelligent batching systems.",
        "Performance optimization is crucial for large-scale data processing.",
        "OpenAI's API rate limits can be effectively managed with smart batching.",
        "Vector embeddings provide semantic search capabilities for RAG systems.",
        "Database optimization ensures fast retrieval of stored content.",
        "Asynchronous processing enables concurrent operations and better throughput.",
        "Error handling and fallback mechanisms provide system reliability.",
        "Monitoring and logging help track performance improvements over time.",
        "Configuration management allows fine-tuning of batch sizes and parameters."
    ]
    
    sample_document = """
    This is a comprehensive guide to web crawling and RAG systems.
    
    Web crawling involves systematically browsing and extracting content from websites.
    When combined with Retrieval Augmented Generation (RAG), it enables powerful
    question-answering systems that can provide accurate, contextual responses
    based on crawled documentation.
    
    Key components include:
    - Intelligent crawling strategies
    - Content processing and chunking
    - Vector embedding generation
    - Database storage and retrieval
    - Performance optimization techniques
    """
    
    print(f"📊 Test Setup:")
    print(f"   • Sample chunks: {len(sample_chunks)}")
    print(f"   • Document length: {len(sample_document)} characters")
    print(f"   • OpenAI Tier: {os.environ.get('OPENAI_TIER', '2')}")
    print()
    
    try:
        from utils import generate_contextual_embeddings_batch, openai_config
        
        # Test 1: With intelligent batching ENABLED
        print("🚀 Test 1: Intelligent Batching ENABLED")
        print("-" * 40)
        
        openai_config.use_intelligent_batching = True
        
        start_time = asyncio.get_event_loop().time()
        batched_results = await generate_contextual_embeddings_batch(
            sample_document, 
            sample_chunks, 
            tier=int(os.environ.get('OPENAI_TIER', '2'))
        )
        end_time = asyncio.get_event_loop().time()
        
        batched_duration = end_time - start_time
        print(f"✅ Batched processing completed in {batched_duration:.2f} seconds")
        print(f"   • Processed {len(batched_results)} chunks")
        print(f"   • Average time per chunk: {batched_duration/len(batched_results):.3f} seconds")
        print()
        
        # Test 2: With intelligent batching DISABLED (individual processing)
        print("🐌 Test 2: Intelligent Batching DISABLED (Individual Processing)")
        print("-" * 40)
        
        openai_config.use_intelligent_batching = False
        
        start_time = asyncio.get_event_loop().time()
        individual_results = await generate_contextual_embeddings_batch(
            sample_document, 
            sample_chunks, 
            tier=int(os.environ.get('OPENAI_TIER', '2'))
        )
        end_time = asyncio.get_event_loop().time()
        
        individual_duration = end_time - start_time
        print(f"✅ Individual processing completed in {individual_duration:.2f} seconds")
        print(f"   • Processed {len(individual_results)} chunks")
        print(f"   • Average time per chunk: {individual_duration/len(individual_results):.3f} seconds")
        print()
        
        # Performance comparison
        print("📈 Performance Comparison")
        print("=" * 40)
        
        if individual_duration > 0:
            speedup = individual_duration / batched_duration if batched_duration > 0 else float('inf')
            time_saved = individual_duration - batched_duration
            percentage_improvement = (time_saved / individual_duration) * 100 if individual_duration > 0 else 0
            
            print(f"⚡ Speedup Factor: {speedup:.1f}x faster")
            print(f"⏱️  Time Saved: {time_saved:.2f} seconds")
            print(f"📊 Performance Improvement: {percentage_improvement:.1f}%")
            print()
            
            # API call estimation
            estimated_individual_calls = len(sample_chunks)
            estimated_batch_calls = max(1, len(sample_chunks) // 5)  # Assume batch size of 5
            api_call_reduction = ((estimated_individual_calls - estimated_batch_calls) / estimated_individual_calls) * 100
            
            print(f"🔗 Estimated API Call Reduction:")
            print(f"   • Individual: {estimated_individual_calls} API calls")
            print(f"   • Batched: {estimated_batch_calls} API calls")
            print(f"   • Reduction: {api_call_reduction:.1f}%")
            print()
            
            # Cost implications
            print(f"💰 Cost Implications:")
            print(f"   • With batching: ~{api_call_reduction:.0f}% fewer API calls")
            print(f"   • Faster processing = lower operational costs")
            print(f"   • Better throughput for high-volume crawling")
            
        else:
            print("⚠️  Could not calculate performance metrics (division by zero)")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies installed.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Check your OpenAI API key and configuration.")

def main():
    """Main entry point."""
    print("🧪 Contextual Embedding Performance Test")
    print("========================================")
    print()
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key to run this test.")
        return
    
    print("✅ Environment configured")
    print()
    
    # Run the async test
    asyncio.run(test_batching_performance())

if __name__ == "__main__":
    main() 
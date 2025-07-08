#!/usr/bin/env python3
"""
Test script for Phase 2: Data extraction, processing, and chunking pipeline.
Demonstrates the complete workflow from Wikipedia extraction to document chunks.
"""

import time
from typing import List
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.logging import setup_logging, logger
from app.services.data_extractor import WikipediaExtractor
from app.services.text_processor import TextProcessor
from app.services.chunking_service import IntelligentChunker
from app.models.documents import RawDocument, ProcessedDocument, DocumentChunk


def test_data_extraction():
    """Test Wikipedia data extraction."""
    print("\nTESTING DATA EXTRACTION")
    print("=" * 50)
    
    try:
        extractor = WikipediaExtractor()
        
        # Get cache info
        cache_info = extractor.get_cache_info()
        print(f"Cache directory: {cache_info['cache_directory']}")
        print(f"Cached files: {cache_info['cached_files']}")
        
        # Extract content - remove section filter to get all content
        print("Extracting content from Wikipedia...")
        start_time = time.time()
        
        # Extract all sections instead of specific ones
        documents = extractor.extract_colombia_content(
            use_cache=True,
            sections=None  # Extract all sections
        )
        
        extraction_time = time.time() - start_time
        
        print(f"Extraction completed in {extraction_time:.2f} seconds")
        print(f"Documents extracted: {len(documents)}")
        
        if documents:
            total_chars = sum(len(doc.content) for doc in documents)
            print(f"Total characters: {total_chars:,}")
            
            # Show sections found
            sections_found = [doc.metadata.section for doc in documents if doc.metadata.section]
            unique_sections = list(set(sections_found))
            print(f"Sections found: {unique_sections}")
            
            print("\nSample documents:")
            for i, doc in enumerate(documents[:3]):  # Show first 3
                section = doc.metadata.section or "Unknown"
                print(f"  {i+1}. {section}: {len(doc.content)} chars")
                print(f"     Preview: {doc.content[:100]}...")
        else:
            print("WARNING: No documents extracted")
        
        return documents
        
    except Exception as e:
        print(f"ERROR: Data extraction failed: {e}")
        return []


def test_text_processing(raw_documents: List[RawDocument]):
    """Test text processing."""
    print("\nTESTING TEXT PROCESSING")
    print("=" * 50)
    
    if not raw_documents:
        print("WARNING: No raw documents to process")
        return []
    
    try:
        processor = TextProcessor()
        
        print(f"Processing {len(raw_documents)} documents...")
        start_time = time.time()
        
        processed_documents, batch_stats = processor.batch_process_documents(
            raw_documents,
            clean_html=True,
            normalize_whitespace=True,
            remove_references=True,
            min_paragraph_length=100,
            require_colombia_relevance=True,
            min_relevance_score=0.1
        )
        
        processing_time = time.time() - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Processing statistics:")
        print(f"   - Input documents: {batch_stats['total_input_documents']}")
        print(f"   - Successfully processed: {batch_stats['successfully_processed']}")
        print(f"   - Failed: {batch_stats['failed_documents']}")
        print(f"   - Skipped (low relevance): {batch_stats['skipped_documents']}")
        print(f"   - Success rate: {batch_stats['success_rate']:.1%}")
        print(f"   - Total processed characters: {batch_stats['total_processed_characters']:,}")
        print(f"   - Total processed words: {batch_stats['total_processed_words']:,}")
        
        if processed_documents:
            # Show processing summary
            summary = processor.get_processing_summary(processed_documents)
            print(f"\nProcessing summary:")
            print(f"   - Average relevance score: {summary['average_relevance_score']:.2f}")
            print(f"   - Average chars per doc: {summary['average_characters_per_doc']:.0f}")
            print(f"   - Content types: {summary['content_types']}")
            print(f"   - Sections: {list(summary['sections'].keys())}")
            
            print("\nSample processed documents:")
            for i, doc in enumerate(processed_documents[:2]):  # Show first 2
                stats = doc.processing_stats
                section = doc.metadata.section or "Unknown"
                print(f"  {i+1}. {section}:")
                print(f"     - Characters: {doc.character_count}")
                print(f"     - Words: {doc.word_count}")
                print(f"     - Relevance: {stats.get('colombia_relevance_score', 0):.2f}")
                print(f"     - Preview: {doc.cleaned_content[:100]}...")
        else:
            print("WARNING: No documents processed successfully")
        
        return processed_documents
        
    except Exception as e:
        print(f"ERROR: Text processing failed: {e}")
        return []


def test_chunking(processed_documents: List[ProcessedDocument]):
    """Test document chunking."""
    print("\nTESTING DOCUMENT CHUNKING")
    print("=" * 50)
    
    if not processed_documents:
        print("WARNING: No processed documents to chunk")
        return []
    
    try:
        chunker = IntelligentChunker()
        
        print(f"Chunking {len(processed_documents)} documents...")
        start_time = time.time()
        
        all_chunks, batch_stats = chunker.batch_chunk_documents(
            processed_documents,
            chunk_size=800,  # Smaller for testing
            chunk_overlap=150,
            preserve_sentences=True,
            min_chunk_size=200,
            max_chunk_size=1200
        )
        
        chunking_time = time.time() - start_time
        
        print(f"Chunking completed in {chunking_time:.2f} seconds")
        print(f"Chunking statistics:")
        print(f"   - Input documents: {batch_stats['total_input_documents']}")
        print(f"   - Successfully chunked: {batch_stats['successfully_chunked']}")
        print(f"   - Failed: {batch_stats['failed_documents']}")
        print(f"   - Total chunks created: {batch_stats['total_chunks_created']}")
        print(f"   - Average chunks per doc: {batch_stats['average_chunks_per_document']:.1f}")
        print(f"   - Average chunk size: {batch_stats['average_chunk_size']:.0f} chars")
        print(f"   - Success rate: {batch_stats['success_rate']:.1%}")
        
        if all_chunks:
            # Show chunking summary
            summary = chunker.get_chunking_summary(all_chunks)
            print(f"\nChunking summary:")
            print(f"   - Total chunks: {summary['total_chunks']}")
            print(f"   - Documents represented: {summary['documents_represented']}")
            print(f"   - Average chunk size: {summary['average_chunk_size']:.0f} chars")
            print(f"   - Average words per chunk: {summary['average_words_per_chunk']:.0f}")
            print(f"   - Size distribution: {summary['size_distribution']}")
            print(f"   - Chunks with overlap: {summary['chunks_with_overlap']}")
            
            print("\nSample chunks:")
            for i, chunk in enumerate(all_chunks[:3]):  # Show first 3
                print(f"  {i+1}. Chunk {chunk.chunk_index} from {chunk.document_id}:")
                print(f"     - Position: {chunk.start_char}-{chunk.end_char}")
                print(f"     - Size: {chunk.character_count} chars, {chunk.word_count} words")
                print(f"     - Overlap: prev={chunk.overlap_with_previous}, next={chunk.overlap_with_next}")
                print(f"     - Content: {chunk.content[:100]}...")
                if chunk.context_before:
                    print(f"     - Context before: {chunk.context_before[:50]}...")
        
        return all_chunks
        
    except Exception as e:
        print(f"ERROR: Chunking failed: {e}")
        return []


def run_complete_pipeline():
    """Run the complete Phase 2 pipeline test."""
    print("COLOMBIA RAG CHATBOT - PHASE 2 PIPELINE TEST")
    print("=" * 60)
    
    overall_start = time.time()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Phase 2 pipeline test")
    
    # Step 1: Data Extraction
    raw_documents = test_data_extraction()
    
    # Step 2: Text Processing
    processed_documents = test_text_processing(raw_documents)
    
    # Step 3: Document Chunking
    chunks = test_chunking(processed_documents)
    
    # Final summary
    total_time = time.time() - overall_start
    
    print("\nPIPELINE TEST COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Final results:")
    print(f"   - Raw documents extracted: {len(raw_documents)}")
    print(f"   - Documents processed: {len(processed_documents)}")
    print(f"   - Chunks created: {len(chunks)}")
    
    if chunks:
        total_content = sum(chunk.character_count for chunk in chunks)
        print(f"   - Total content: {total_content:,} characters")
        print(f"   - Average chunk size: {total_content / len(chunks):.0f} characters")
        
        # Success metrics
        extraction_success = len(raw_documents) > 0
        processing_success = len(processed_documents) > 0
        chunking_success = len(chunks) > 0
        
        print(f"\nSuccess indicators:")
        print(f"   - Data extraction: {'PASS' if extraction_success else 'FAIL'}")
        print(f"   - Text processing: {'PASS' if processing_success else 'FAIL'}")
        print(f"   - Document chunking: {'PASS' if chunking_success else 'FAIL'}")
        
        overall_success = extraction_success and processing_success and chunking_success
        print(f"\nOverall pipeline: {'SUCCESS' if overall_success else 'FAILED'}")
        
        if overall_success:
            print("\nPhase 2 implementation is working correctly!")
            print("Ready to proceed with Phase 3: RAG System Implementation")
        else:
            print("\nSome components need attention before proceeding.")
    else:
        print("\nPipeline failed - no chunks were created")
    
    logger.info("Phase 2 pipeline test completed", total_time=total_time)


if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
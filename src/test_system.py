# test_system.py

"""
Test script to verify all components are working properly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all components can be imported"""
    print("🧪 Testing component imports...")
    
    try:
        from document_processor import EnhancedDocumentProcessor
        print("✅ Document processor imported successfully")
    except ImportError as e:
        print(f"❌ Document processor import failed: {e}")
        return False
    
    try:
        from image_processor import TEPPLImageProcessor, TEPPLSmartImageFilter
        print("✅ Image processor imported successfully")
    except ImportError as e:
        print(f"❌ Image processor import failed: {e}")
        return False
    
    try:
        from chroma_multimodal_store import EnhancedMultimodalVectorStore
        print("✅ Vector store imported successfully")
    except ImportError as e:
        print(f"❌ Vector store import failed: {e}")
        return False
    
    try:
        from batch_processor import EnhancedBatchProcessor
        print("✅ Batch processor imported successfully")
    except ImportError as e:
        print(f"❌ Batch processor import failed: {e}")
        return False
    
    return True

def test_image_processor():
    """Test the image processor functionality"""
    print("\n🎨 Testing image processor functionality...")
    
    try:
        from image_processor import TEPPLImageProcessor, TEPPLSmartImageFilter
        
        # Test smart filter
        filter_instance = TEPPLSmartImageFilter()
        print("✅ Smart filter initialized")
        
        # Test image processor
        processor = TEPPLImageProcessor()
        print("✅ Image processor initialized")
        
        # Test filtering with dummy data
        dummy_images = [
            {
                'dimensions': {'width': 200, 'height': 200},
                'size_bytes': 10000,
                'extracted_text': 'traffic sign regulatory warning',
                'context': {'nearby_text': 'speed limit 25 mph'},
                'type': 'embedded_image'
            },
            {
                'dimensions': {'width': 50, 'height': 50},
                'size_bytes': 1000,
                'extracted_text': 'ncdot logo',
                'context': {'nearby_text': 'department of transportation'},
                'type': 'embedded_image'
            }
        ]
        
        filtered = filter_instance.filter_images(dummy_images)
        print(f"✅ Filtering test: {len(dummy_images)} → {len(filtered)} images")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processor test failed: {e}")
        return False

def test_vector_store():
    """Test the vector store functionality"""
    print("\n💾 Testing vector store functionality...")
    
    try:
        from chroma_multimodal_store import EnhancedMultimodalVectorStore
        
        # Initialize vector store
        vector_store = EnhancedMultimodalVectorStore("./test_storage")
        print("✅ Vector store initialized")
        
        # Test getting stats
        stats = vector_store.get_enhanced_collection_stats()
        print(f"✅ Collection stats retrieved: {stats['total_items']} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_document_processor():
    """Test the document processor functionality"""
    print("\n📄 Testing document processor functionality...")
    
    try:
        from document_processor import EnhancedDocumentProcessor
        
        # Initialize processor
        processor = EnhancedDocumentProcessor()
        print("✅ Document processor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_batch_processor():
    """Test the batch processor functionality"""
    print("\n📦 Testing batch processor functionality...")
    
    try:
        from batch_processor import EnhancedBatchProcessor
        
        # Initialize batch processor
        processor = EnhancedBatchProcessor()
        print("✅ Batch processor initialized")
        
        # Test discovering files
        files = processor.discover_all_files()
        print(f"✅ File discovery: {len(files)} files found")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting TEPPL System Component Tests")
    print("="*50)
    
    tests = [
        test_imports,
        test_image_processor,
        test_vector_store,
        test_document_processor,
        test_batch_processor
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    print(f"❌ Failed: {total - passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    print("\n📋 NEXT STEPS:")
    print("1. Run: python batch_processor_complete.py --dry-run")
    print("2. If dry-run looks good, run: python batch_processor_complete.py")
    print("3. Monitor progress with: python batch_processor_complete.py --status")

if __name__ == "__main__":
    main()
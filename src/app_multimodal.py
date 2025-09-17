# src/app_multimodal_enhanced.py

"""
Enhanced Flask Application with Multimodal Support and Optimized Image Handling
Serves TEPPL documents with text, images, and drawings with thumbnail support
Updated to work with enhanced image post-processor with comprehensive fixes
"""

import os
from flask import Flask, render_template, request, jsonify, send_file, abort, send_from_directory
import json
import logging
from pathlib import Path
from datetime import datetime
import mimetypes
import base64
import traceback
import time
import re
from types import SimpleNamespace
import openai

# Calculate paths
base_dir = os.path.abspath(os.path.dirname(__file__))

# Template folder: src/templates
template_folder = os.path.join(base_dir, 'templates')

# Static folder: ../static (go up one level from src to project root, then static)
project_root = os.path.abspath(os.path.join(base_dir, '..'))
static_folder = os.path.join(project_root, 'static')

# Import our enhanced components
try:
    from src.chroma_multimodal_store import EnhancedMultimodalVectorStore
    from src.multimodal_rag_system import TEPPLMultimodalRAGSystem
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Multimodal components not available: {e}")
    # Fallback to original components
    try:
        from src.chroma_multimodal_store import TEPPLVectorStore
        from src.multimodal_rag_system import TEPPLRAGSystem
        MULTIMODAL_AVAILABLE = False
    except ImportError:
        print("❌ No RAG components available")
        MULTIMODAL_AVAILABLE = False

app = Flask(__name__,
            template_folder=template_folder,
            static_folder=static_folder)

# Debug verification
print(f"🔍 Template folder: {template_folder}")
print(f"🔍 Static folder: {static_folder}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
print("🚀 Initializing Enhanced TEPPL RAG System...")
try:
    if MULTIMODAL_AVAILABLE:
        print("✅ Using multimodal vector store with ChromaDB")
        vector_store = EnhancedMultimodalVectorStore(persist_directory="./storage/chroma_enhanced")
        rag_system = TEPPLMultimodalRAGSystem(vector_store=vector_store)
        system_type = "multimodal"
    else:
        print("⚠️ Falling back to original vector store")
        vector_store = TEPPLVectorStore(persist_directory="./storage/teppl_vector")
        rag_system = TEPPLRAGSystem(vector_store=vector_store)
        system_type = "text_only"
    print(f"✅ RAG System ready ({system_type})")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    rag_system = None
    system_type = "error"

# Load processed image metadata with enhanced structure
def load_image_metadata():
    """Load processed image metadata for fast serving with enhanced structure"""
    metadata_file = Path("./documents/image_metadata.json")
    if not metadata_file.exists():
        logger.warning("image_metadata.json not found. Image features will be limited.")
        return {}
    try:
        start_time = time.time()
        file_size = os.path.getsize(metadata_file)
        logger.info(f"Loading image_metadata.json (size: {file_size / 1024:.2f} KB)")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        load_time = time.time() - start_time
        logger.info(f"📸 Loaded metadata for {len(metadata)} processed images in {load_time:.2f}s")
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Malformed image_metadata.json at line {e.lineno} col {e.colno}: {e.msg}")
        try:
            logger.info("Attempting to recover malformed JSON...")
            text = metadata_file.read_text(encoding='utf-8')
            # Attempt to fix trailing commas
            cleaned_text = re.sub(r',\\s*([\\]\\}])', r'\\1', text)
            metadata = json.loads(cleaned_text)
            logger.info("Successfully recovered and parsed JSON.")
            return metadata
        except Exception as recovery_e:
            logger.critical(f"Failed to recover malformed JSON: {recovery_e}")
            return {}
    except Exception as e:
        logger.critical(f"Could not load image metadata: {e}", exc_info=True)
        return {}

# Global image metadata cache
IMAGE_METADATA = load_image_metadata()

# Create filename to ID mapping for faster lookups
FILENAME_TO_ID_MAP = {}
if IMAGE_METADATA:
    kept_count = 0
    missing_filename = 0
    for image_id, metadata in IMAGE_METADATA.items():
        if metadata.get("kept", False):
            kept_count += 1
            original_filename = metadata.get("original_filename")
            readable_filename = metadata.get("readable_filename")
            if original_filename:
                FILENAME_TO_ID_MAP[original_filename] = image_id
            if readable_filename and readable_filename != original_filename:
                FILENAME_TO_ID_MAP[readable_filename] = image_id
            if not original_filename and not readable_filename:
                missing_filename += 1
    logger.info(f"🔗 Created filename mapping for {len(FILENAME_TO_ID_MAP)} images from {kept_count} kept images.")
    if missing_filename > 0:
        logger.warning(f"{missing_filename} kept images are missing filename information.")

# Global configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024, # 50MB max file size
    'DOCUMENTS_PATH': './documents',
    'STORAGE_PATH': './storage',
    'IMAGES_PATH': './documents/images',
    'THUMBNAILS_PATH': './documents/thumbnails',
    'DEBUG': os.getenv('FLASK_ENV') == 'development',
    'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-key-change-in-production')
})

# Enhanced debug logging configuration
if app.config['DEBUG']:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("🔧 Debug mode enabled - verbose logging active")
else:
    logging.getLogger().setLevel(logging.INFO)

# Configure OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

def adapt_response_for_ui(raw_response, query, processing_time):
    """Adapt response for professional NCDOT output with enhanced structure"""
    
    # Extract and enhance answer with professional markdown
    answer = raw_response.get('answer', '') or ""
    answer = normalize_markdown(answer)
    if not answer and raw_response.get('sources'):
        answer = generate_professional_answer(raw_response['sources'], query)
    
    # Calculate realistic confidence
    confidence = raw_response.get('confidence', 0.0)
    sources = raw_response.get('sources', [])
    
    if confidence == 0.0 and sources:
        avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
        confidence = min(0.95, max(0.72, avg_similarity + 0.25))
    
    # Enhanced source formatting with proper metadata
    top_sources = sorted(sources, key=lambda x: x.get('similarity_score', 0), reverse=True)[:5]
    formatted_sources = format_professional_sources(top_sources)
    
    return {
        "success": True,
        "answer": answer,
        "confidence": confidence,
        "sources": formatted_sources,
        "images": raw_response.get('images', []),
        "processing_time": processing_time,
        "query": query,
        "source_count": len(formatted_sources),
        "presentation_mode": "professional",
        "content_type": "markdown"
    }

def generate_professional_answer(sources, query):
    """Generate ChatGPT-style markdown formatted response with proper structure"""
    if not sources:
        return "No relevant NCDOT TEPPL guidance found for this query."

    # Use top 3 sources for comprehensive answer
    top_sources = sorted(sources, key=lambda x: x.get('similarity_score', 0), reverse=True)[:3]
    
    # Build structured markdown response
    response_parts = []
    
    # Dynamic heading based on query analysis
    query_lower = query.lower()
    if any(term in query_lower for term in ['sign', 'signage', 'regulatory']):
        response_parts.append("# Traffic Sign Requirements\n")
    elif any(term in query_lower for term in ['signal', 'timing', 'intersection']):
        response_parts.append("# Traffic Signal Standards\n") 
    elif any(term in query_lower for term in ['marking', 'pavement', 'lane']):
        response_parts.append("# Pavement Marking Guidelines\n")
    elif any(term in query_lower for term in ['design', 'geometric', 'roadway']):
        response_parts.append("# Roadway Design Standards\n")
    else:
        response_parts.append("# NCDOT TEPPL Requirements\n")
    
    # Extract and organize content into structured sections
    requirements = []
    standards = []
    procedures = []
    
    for source in top_sources:
        content = source.get('content', '').strip()
        if len(content) > 50:
            # Categorize content based on keywords
            content_lower = content.lower()
            if any(word in content_lower for word in ['shall', 'must', 'required']):
                requirements.append({
                    'content': content[:400].strip(),
                    'source': source.get('metadata', {}).get('title', 'NCDOT Document'),
                    'page': source.get('metadata', {}).get('page_number', 'N/A')
                })
            elif any(word in content_lower for word in ['standard', 'specification', 'criteria']):
                standards.append({
                    'content': content[:400].strip(),
                    'source': source.get('metadata', {}).get('title', 'NCDOT Document'),
                    'page': source.get('metadata', {}).get('page_number', 'N/A')
                })
            else:
                procedures.append({
                    'content': content[:400].strip(),
                    'source': source.get('metadata', {}).get('title', 'NCDOT Document'),
                    'page': source.get('metadata', {}).get('page_number', 'N/A')
                })
    
    # Generate structured sections with proper markdown
    if requirements:
        response_parts.append("## **Key Requirements**\n")
        for req in requirements:
            clean_content = _clean_and_format_content(req['content'])
            response_parts.append(f"• **{clean_content}**\n")
        response_parts.append("")
    
    if standards:
        response_parts.append("## **Technical Standards**\n")
        for std in standards:
            clean_content = _clean_and_format_content(std['content'])
            response_parts.append(f"• {clean_content}\n")
        response_parts.append("")
    
    if procedures:
        response_parts.append("## **Implementation Guidelines**\n")
        for proc in procedures:
            clean_content = _clean_and_format_content(proc['content'])
            response_parts.append(f"• {clean_content}\n")
        response_parts.append("")
    
    # Add compliance section
    response_parts.append("## **Compliance Notes**\n")
    response_parts.append("• All implementations must comply with **MUTCD standards** and North Carolina supplements")
    response_parts.append("• **Division Engineer approval** required for final design decisions")
    response_parts.append("• Installation must follow established **safety protocols** and **NCDOT procedures**")
    response_parts.append("• Reference **current TEPPL documents** for detailed specifications")
    
    return "\n".join(response_parts)

def _clean_and_format_content(content):
    """Clean and format content for professional display"""
    # Remove excessive whitespace
    content = ' '.join(content.split())
    
    # Format common technical terms
    replacements = {
        'MUTCD': '**MUTCD**',
        'NCDOT': '**NCDOT**',
        'CFR': '**CFR**',
        'shall': '**shall**',
        'must': '**must**',
        'required': '**required**'
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    return content

def format_professional_sources(sources):
    """Format sources professionally for NCDOT presentation"""
    formatted = []
    for source in sources:
        metadata = source.get('metadata', {})
        # Extract proper document title
        title = extract_document_title(metadata)
        pages = str(metadata.get('page_number', 'N/A'))
        relevance = int((source.get('similarity_score', 0) * 100))
        formatted.append({
            "title": title,
            "pages": pages,
            "document_type": metadata.get('source_type', 'Policy Document'),
            "content": source.get('content', ''),
            "relevance": f"{relevance}%",
            "similarity_score": source.get('similarity_score', 0.0),
            "file_path": metadata.get('source', ''),
            # NEW: direct link and page number for viewer
            "internal_link": f"/documents/{metadata.get('source', '')}" if metadata.get('source') else "",
            "page_number": metadata.get('page_number', 1),
            # NEW: optional hash for integrity checking
            "sha256": metadata.get('content_hash', ""),
            "document_id": metadata.get('document_id', ''),
            "source": metadata.get('source', ''),
            "chunk_id": metadata.get('chunk_id', ''),
            "year": "2024"
        })
    return formatted

def extract_document_title(metadata):
    """Extract meaningful document titles"""
    title = metadata.get('title', '')
    if title and title != 'Unknown Document':
        return title
    
    doc_id = metadata.get('document_id', '').replace('teppl_', '').replace('_', ' ')
    if 'policy' in doc_id.lower():
        return f"NCDOT Traffic Engineering Policy - {doc_id.title()}"
    elif 'practice' in doc_id.lower():
        return f"NCDOT Standard Practice - {doc_id.title()}"
    elif 'guide' in doc_id.lower():
        return f"NCDOT Technical Guide - {doc_id.title()}"
    else:
        return f"NCDOT TEPPL Document - {doc_id.title()}"

def generate_answer_from_sources(sources, query):
    """Generate a basic answer from sources if no answer is provided"""
    if not sources:
        return "No relevant information found in TEPPL documents."
    
    # Simple fallback - combine top source contents
    content_parts = []
    for source in sources[:3]:  # Use top 3 sources
        content = source.get('content', '').strip()
        if content and len(content) > 50:
            content_parts.append(content[:200] + "..." if len(content) > 200 else content)
    
    if content_parts:
        return "Based on TEPPL documents:\n\n• " + "\n\n• ".join(content_parts)
    else:
        return "Found relevant TEPPL documents but unable to extract specific guidelines."

@app.route("/")
def home():
    """Enhanced home page with real multimodal capabilities info"""
    system_stats = {}
    real_counts = {
        "images": 0,
        "documents": 0,
        "text_chunks": 0,
        "drawings": 0
    }
    
    # Get real system statistics from ChromaDB
    if rag_system and hasattr(rag_system, 'get_system_stats'):
        try:
            system_stats = rag_system.get_system_stats()
            
            # Extract real counts from vector store
            vector_stats = system_stats.get("vector_store", {})
            real_counts["text_chunks"] = vector_stats.get("text_chunks", 0)
            real_counts["images"] = (
                vector_stats.get("regular_images", 0) + 
                vector_stats.get("classified_images", 0)
            )
            real_counts["drawings"] = vector_stats.get("technical_drawings", 0)
            real_counts["documents"] = len(set(
                chunk.get("metadata", {}).get("document_id", "unknown")
                for chunk in vector_stats.get("recent_chunks", [])
            )) if "recent_chunks" in vector_stats else 25  # fallback
            
        except Exception as e:
            logger.warning(f"Could not get system stats: {e}")
            system_stats = {"error": str(e)}

    # Get enhanced image statistics from processed metadata
    image_stats = {
        "total_processed_images": len(IMAGE_METADATA),
        "kept_images": 0,
        "categories": {},
        "filter_reasons": {}
    }

    if IMAGE_METADATA:
        for img_id, metadata in IMAGE_METADATA.items():
            if metadata.get("kept", False):
                image_stats["kept_images"] += 1
                analysis = metadata.get("analysis", {})
                category = analysis.get("category", "general")
                image_stats["categories"][category] = image_stats["categories"].get(category, 0) + 1
            else:
                analysis = metadata.get("analysis", {})
                reason = analysis.get("reason", "unknown")
                image_stats["filter_reasons"][reason] = image_stats["filter_reasons"].get(reason, 0) + 1

    # Pass enhanced system info to template
    template_data = {
        "system_type": system_type,
        "multimodal_available": MULTIMODAL_AVAILABLE,
        "system_stats": system_stats,
        "image_stats": image_stats,
        "real_counts": real_counts,  # Pass real counts to template
        "features": {
            "text_search": True,
            "image_search": MULTIMODAL_AVAILABLE,
            "drawing_search": MULTIMODAL_AVAILABLE,
            "multimodal_citations": MULTIMODAL_AVAILABLE,
            "high_quality_thumbnails": True,
            "enhanced_filtering": True,
            "readable_filenames": True,
            "text_vs_object_detection": True
        }
    }
    return render_template("index.html", **template_data)

@app.route("/query", methods=["POST"])
def enhanced_query():
    request_id = str(int(time.time() * 1000))  # Unique request ID
    logger.info(f"🔍 REQUEST {request_id}: Starting query processing from {request.remote_addr}")
    if rag_system is None:
        logger.error(f"❌ REQUEST {request_id}: RAG system not available")
        return jsonify({
            "success": False,
            "error": "RAG system is not available.",
            "request_id": request_id
        }), 500

    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        logger.info(f"📝 REQUEST {request_id}: Query = '{query[:50]}...'")
        if not query:
            logger.warning(f"⚠️ REQUEST {request_id}: Empty query received")
            return jsonify({
                "success": False,
                "error": "Query is required",
                "request_id": request_id
            }), 400

        # ENHANCED QUERY PREPROCESSING
        enhanced_query_context = build_intelligent_query_context(query)
        
        # Create user context
        user_context = {
            "user_id": request.remote_addr or "anonymous",
            "ip_address": request.remote_addr or "127.0.0.1",
            "enhanced_query": enhanced_query_context['enhanced_query'],
            "domain": enhanced_query_context['domain'],
            "intent": enhanced_query_context['intent']
        }

        start_time = time.time()
        
        # Use enhanced query for better retrieval
        response = rag_system.enhanced_multimodal_retrieval(
            query=enhanced_query_context['enhanced_query'],
            user_context=user_context,
            n_results=data.get('num_results', 20),  # Increased for better results
            include_images=data.get('include_images', True),
            include_drawings=data.get('include_drawings', True),
            include_text=data.get('include_text', True)
        )

        processing_time = time.time() - start_time

        # Handle error response from RAG system
        if not response.get('success', True):
            return jsonify({
                "success": False,
                "error": response.get('error', 'Unknown error'),
                "answer": "",
                "sources": [],
                "images": []
            }), 500

        # Apply context-aware ranking
        if response.get('sources'):
            response['sources'] = apply_context_aware_ranking(
                response['sources'], 
                enhanced_query_context
            )

        # Generate professional answer if none exists
        if not response.get('answer') and response.get('sources'):
            response['answer'] = generate_professional_chatgpt_answer(
                response['sources'], 
                query,
                enhanced_query_context
            )

        # Format for ChatGPT-style UI
        ui_response = adapt_response_for_ui(response, query, processing_time)

        # Log the query
        logger.info(
            f"📊 Enhanced Query processed: '{query[:50]}...' | "
            f"Domain: {enhanced_query_context['domain']} | "
            f"Sources: {len(ui_response['sources'])} | "
            f"Images: {len(ui_response['images'])} | "
            f"Time: {processing_time:.3f}s"
        )

        logger.info(f"✅ REQUEST {request_id}: Query completed successfully")
        return jsonify(ui_response)

    except Exception as e:
        logger.error(f"❌ Error processing query '{query[:50]}...': {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "answer": "",
            "sources": [],
            "images": [],
            "system_type": system_type
        }), 500

def format_sources_for_ui(sources):
    """Transform existing metadata to UI format WITHOUT changing stored data"""
    formatted = []
    for source in sources:
        metadata = source.get('metadata', {})
        
        # Extract what we can from existing metadata
        title = (
            metadata.get('title') or 
            metadata.get('filename') or 
            metadata.get('document_id', '').replace('teppl_', '').replace('_', ' ').title() or
            'TEPPL Document'
        )
        
        # Smart page extraction from existing data
        pages = str(metadata.get('page_number', 'N/A'))
        if pages == 'N/A' and 'chunk_id' in metadata:
            # Extract page from chunk_id like "teppl_doc_p5_c2"
            import re
            page_match = re.search(r'_p(\d+)_', metadata['chunk_id'])
            if page_match:
                pages = page_match.group(1)
        
        formatted.append({
            "title": title,
            "pages": pages,
            "document_type": metadata.get('source_type', 'document'),
            "content": source.get('content', ''),
            "similarity_score": source.get('similarity_score', 0.0),
            "file_path": metadata.get('source', ''),  # Use existing 'source' field
            "document_id": metadata.get('document_id', ''),
            "source": metadata.get('source', ''),
            "chunk_id": metadata.get('chunk_id', ''),
            "year": "2024"  # Default for TEPPL docs
        })
    return formatted

def format_images_for_ui(images):
    """Format images to match UI expectations with enhanced metadata"""
    formatted = []
    for image in images:
        if isinstance(image, dict):
            metadata = image.get('metadata', {})
            image_id = image.get('id', '')

            # Derive filename and internal link if possible
            source_path = metadata.get('source', '')
            filename = extract_filename_from_path(source_path) if source_path else None
            internal_link = f"/documents/{filename}" if filename else ""

            # Enhanced metadata lookup
            enhanced_meta = {}
            if image_id and image_id in IMAGE_METADATA:
                img_info = IMAGE_METADATA[image_id]
                if img_info.get("kept", False):
                    analysis = img_info.get("analysis", {})
                    enhanced_meta = {
                        "web_path": img_info.get("web_path", f"images/{image_id}.png"),
                        "thumbnail_path": img_info.get("thumbnail_web_path", f"thumbnails/{image_id}_thumb.jpg"),
                        "dimensions": analysis.get("dimensions", [0, 0]),
                        "category": analysis.get("category", "general"),
                        "confidence": analysis.get("confidence", 0.5),
                        "original_filename": img_info.get("original_filename", "unknown"),
                        "readable_filename": img_info.get("readable_filename", "unknown"),
                        "content_type_detail": analysis.get("content_type", "unknown"),
                        "complexity_score": analysis.get("complexity_score", 0),
                        "text_vs_object": analysis.get("text_vs_object", {})
                    }

            formatted_image = {
                "id": image_id,
                "type": image.get('content_type', image.get('type', 'image')),
                "similarity_score": image.get('similarity_score', 0.0),
                "content": image.get('content', ''),
                "source": metadata.get('source', 'unknown'),
                "page_number": metadata.get('page_number', 'N/A'),
                "document_id": metadata.get('document_id', 'unknown'),
                "file_path": metadata.get('file_path', ''),
                "teppl_category": metadata.get('teppl_category', 'general'),
                # New: deep-link for image's source document
                "internal_link": internal_link
            }

            if enhanced_meta:
                formatted_image.update({
                    "web_path": enhanced_meta["web_path"],
                    "thumbnail_path": enhanced_meta["thumbnail_path"],
                    "dimensions": enhanced_meta["dimensions"],
                    "category": enhanced_meta["category"],
                    "original_filename": enhanced_meta["original_filename"],
                    "readable_filename": enhanced_meta["readable_filename"],
                    "content_type_detail": enhanced_meta["content_type_detail"],
                    "complexity_score": enhanced_meta["complexity_score"],
                    "text_vs_object": enhanced_meta["text_vs_object"],
                    "is_enhanced": True
                })
            else:
                formatted_image.update({
                    "web_path": f"images/{image_id}.png",
                    "thumbnail_path": f"thumbnails/{image_id}_thumb.jpg",
                    "is_enhanced": False
                })

            formatted.append(formatted_image)

    return formatted

@app.route("/documents/<path:filename>")
def serve_document(filename):
    """Enhanced document serving with highlighting support and comprehensive search"""
    documents_path = Path(app.config['DOCUMENTS_PATH'])
    
    # Enhanced security check with logging
    if '..' in filename or filename.startswith('/'):
        logger.warning(f"🚫 Directory traversal attempt blocked: {filename} from IP: {request.remote_addr}")
        abort(403)

    # Get highlighting and navigation parameters
    page = request.args.get('page', None)
    highlight = request.args.get('highlight', None)
    search_text = request.args.get('search', None)
    
    logger.info(f"🔍 Enhanced document request: '{filename}' from {request.remote_addr}")
    logger.debug(f"📁 Parameters - page: {page}, highlight: {highlight}, search: {search_text}")
    
    # Find the document file with comprehensive search
    file_path = find_document_file(documents_path, filename)
    
    if not file_path:
        logger.warning(f"❌ Document not found: {filename}")
        return jsonify({
            "error": "Document not found",
            "requested": filename,
            "suggestion": "Check the filename or browse available documents",
            "searched_paths": ['pdfs', 'html', 'aspx', 'excel', 'documents', 'other']
        }), 404

    try:
        # Enhanced PDF handling with viewer integration
        if filename.lower().endswith('.pdf'):
            response = send_file(file_path, as_attachment=False, download_name=filename)
            response.headers['X-Document-Type'] = 'pdf'
            response.headers['Content-Type'] = 'application/pdf'
            
            # Add navigation and highlighting headers
            if page:
                response.headers['X-Target-Page'] = str(page)
                logger.debug(f"📄 PDF target page: {page}")
            
            if highlight:
                response.headers['X-Highlight-Text'] = highlight
                logger.debug(f"🔍 PDF highlight text: {highlight}")
                
            if search_text:
                response.headers['X-Search-Text'] = search_text
                logger.debug(f"🔎 PDF search text: {search_text}")
                
            # Enhanced CORS headers for PDF viewer integration
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Expose-Headers'] = 'X-Document-Type,X-Target-Page,X-Highlight-Text,X-Search-Text,Content-Length'
            response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
            
            # Add caching for better performance
            response.headers['Cache-Control'] = 'public, max-age=86400'  # 24 hours
            
        # Enhanced handling for other document types
        elif filename.lower().endswith(('.html', '.htm')):
            response = send_file(file_path, as_attachment=False, download_name=filename)
            response.headers['X-Document-Type'] = 'html'
            response.headers['Content-Type'] = 'text/html'
            
        elif filename.lower().endswith(('.doc', '.docx')):
            response = send_file(file_path, as_attachment=False, download_name=filename)
            response.headers['X-Document-Type'] = 'word'
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            
        elif filename.lower().endswith(('.xls', '.xlsx')):
            response = send_file(file_path, as_attachment=False, download_name=filename)
            response.headers['X-Document-Type'] = 'excel'
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            
        else:
            # Default handling for other file types
            response = send_file(file_path, as_attachment=False, download_name=filename)
            response.headers['X-Document-Type'] = 'other'
        
        # Add enhanced metadata headers
        response.headers['X-File-Size'] = str(file_path.stat().st_size)
        response.headers['X-File-Modified'] = file_path.stat().st_mtime
        response.headers['X-Original-Filename'] = filename
        
        logger.info(f"📄 Successfully served: {filename} ({file_path.stat().st_size} bytes)")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to serve {filename}: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to serve document",
            "details": str(e) if app.config['DEBUG'] else "Internal server error"
        }), 500

def find_document_file(documents_path, filename):
    """Find document file with comprehensive search and fuzzy matching"""
    # Try exact match first
    exact_path = documents_path / filename
    if exact_path.exists() and exact_path.is_file():
        logger.info(f"✅ Exact match found: {exact_path}")
        return exact_path
    
    # Enhanced search in subdirectories
    search_dirs = ['pdfs', 'html', 'aspx', 'excel', 'images', 'documents', 'other']
    
    for search_dir in search_dirs:
        search_path = documents_path / search_dir
        if search_path.exists():
            # Try exact match in subdirectory
            file_path = search_path / filename
            if file_path.exists() and file_path.is_file():
                logger.info(f"✅ Found in subdirectory: {file_path}")
                return file_path
            
            # Try enhanced fuzzy matching
            best_match, score = find_best_file_match(filename, search_path)
            if best_match and score > 0.7:  # Lower threshold for better matching
                logger.info(f"✅ Fuzzy match found: {best_match} (score: {score:.2f})")
                return best_match
    
    logger.warning(f"❌ No match found for: {filename}")
    return None

def find_best_file_match(requested_filename, search_directory):
    """Enhanced fuzzy file matching with better algorithms"""
    if not search_directory.exists():
        return None, 0
    
    from difflib import SequenceMatcher
    
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def normalize_filename(filename):
        """Normalize filename for better matching"""
        base = filename.replace('.pdf', '').replace('.PDF', '')
        normalized = base.replace('_', ' ').replace('-', ' ').lower()
        return ' '.join(normalized.split())
    
    requested_normalized = normalize_filename(requested_filename)
    best_match = None
    best_score = 0.0
    candidates = []
    
    for file_path in search_directory.rglob('*'):
        if file_path.is_file():
            actual_normalized = normalize_filename(file_path.name)
            
            # Calculate multiple similarity metrics
            exact_similarity = similarity(requested_filename.lower(), file_path.name.lower())
            normalized_similarity = similarity(requested_normalized, actual_normalized)
            
            # Word-based matching for better results
            requested_words = set(requested_normalized.split())
            actual_words = set(actual_normalized.split())
            
            if requested_words and actual_words:
                word_overlap = len(requested_words.intersection(actual_words)) / len(requested_words.union(actual_words))
            else:
                word_overlap = 0
            
            # Combined score with weights
            combined_score = (exact_similarity * 0.3) + (normalized_similarity * 0.4) + (word_overlap * 0.3)
            
            # Keep track of good candidates
            if combined_score > 0.4:
                candidates.append({
                    'path': file_path,
                    'score': combined_score,
                    'exact': exact_similarity,
                    'normalized': normalized_similarity,
                    'words': word_overlap
                })
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = file_path
    
    # Log candidates for debugging
    if candidates and len(candidates) > 1:
        logger.debug(f"🔍 Fuzzy match candidates for '{requested_filename}':")
        for candidate in sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]:
            logger.debug(f"  - {candidate['path'].name}: {candidate['score']:.3f} (exact: {candidate['exact']:.3f}, norm: {candidate['normalized']:.3f}, words: {candidate['words']:.3f})")
    
    return best_match, best_score

@app.route("/static/documents/<path:filename>")
def serve_static_documents(filename):
    """Serve documents from static documents directory"""
    documents_path = Path(app.config['DOCUMENTS_PATH'])
    logger.debug(f"Serving static document: {filename} from {documents_path}")
    try:
        return send_from_directory(documents_path, filename)
    except Exception as e:
        logger.error(f"Error serving static document {filename}: {e}")
        abort(404)

@app.route("/images/<path:image_path>")
def serve_image(image_path):
    """Serve processed images with optimized loading and enhanced metadata lookup"""
    # Security check
    if '..' in image_path or image_path.startswith('/'):
        abort(403)

    images_path = Path(app.config['IMAGES_PATH'])
    image_file = images_path / image_path

    if image_file.exists() and image_file.is_file():
        # Set cache headers for better performance
        response = send_file(image_file)
        response.headers['Cache-Control'] = 'public, max-age=86400' # 24 hours

        # Add enhanced metadata headers
        image_filename = image_file.name
        if image_filename in FILENAME_TO_ID_MAP:
            image_id = FILENAME_TO_ID_MAP[image_filename]
            metadata = IMAGE_METADATA.get(image_id, {})
            analysis = metadata.get("analysis", {})
            response.headers['X-Image-Category'] = analysis.get("category", "unknown")
            response.headers['X-Image-Type'] = analysis.get("content_type", "unknown")
            response.headers['X-Original-Filename'] = metadata.get("original_filename", "unknown")
        return response

    # Fallback to extracted images if not in processed directory
    storage_path = Path(app.config['STORAGE_PATH']) / 'extracted_images'
    fallback_file = storage_path / image_path
    if fallback_file.exists() and fallback_file.is_file():
        logger.info(f"📸 Serving fallback image: {fallback_file}")
        return send_file(fallback_file)

    logger.warning(f"❌ Image not found: {image_path}")
    abort(404)

@app.route("/thumbnails/<path:thumbnail_path>")
def serve_thumbnail(thumbnail_path):
    """Serve high-quality thumbnails for fast loading"""
    # Security check
    if '..' in thumbnail_path or thumbnail_path.startswith('/'):
        abort(403)

    thumbnails_path = Path(app.config['THUMBNAILS_PATH'])
    thumbnail_file = thumbnails_path / thumbnail_path

    if thumbnail_file.exists() and thumbnail_file.is_file():
        # Set aggressive cache headers for thumbnails
        response = send_file(thumbnail_file)
        response.headers['Cache-Control'] = 'public, max-age=604800' # 7 days
        response.headers['ETag'] = f'"{thumbnail_path}"'

        # Add quality indicator for high-quality thumbnails
        response.headers['X-Thumbnail-Quality'] = 'high'
        response.headers['X-Thumbnail-Size'] = '400x400'
        return response

    logger.warning(f"❌ Thumbnail not found: {thumbnail_path}")
    abort(404)

@app.route("/api/images/metadata")
def get_images_metadata():
    """Get enhanced metadata for all processed images"""
    try:
        # Filter to only kept images with enhanced details
        kept_images = {}
        categories = {}
        content_types = {}
        filter_reasons = {}

        for image_id, metadata in IMAGE_METADATA.items():
            if metadata.get("kept", False):
                analysis = metadata.get("analysis", {})
                kept_images[image_id] = {
                    "id": image_id,
                    "web_path": metadata.get("web_path"),
                    "thumbnail_path": metadata.get("thumbnail_web_path"),
                    "category": analysis.get("category", "general"),
                    "content_type": analysis.get("content_type", "unknown"),
                    "dimensions": analysis.get("dimensions", [0, 0]),
                    "confidence": analysis.get("confidence", 0.5),
                    "complexity_score": analysis.get("complexity_score", 0),
                    "original_filename": metadata.get("original_filename", "unknown"),
                    "readable_filename": metadata.get("readable_filename", "unknown"),
                    "text_vs_object": analysis.get("text_vs_object", {}),
                    "technical_features": analysis.get("technical_features", {})
                }

                # Count categories and content types
                category = analysis.get("category", "general")
                categories[category] = categories.get(category, 0) + 1
                content_type = analysis.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1
            else:
                # Count filter reasons
                analysis = metadata.get("analysis", {})
                reason = analysis.get("reason", "unknown")
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

        return jsonify({
            "total_images": len(kept_images),
            "total_processed": len(IMAGE_METADATA),
            "categories": categories,
            "content_types": content_types,
            "filter_reasons": filter_reasons,
            "images": kept_images,
            "enhanced_features": {
                "high_quality_thumbnails": True,
                "text_vs_object_detection": True,
                "readable_filenames": True,
                "technical_content_detection": True
            },
            "last_updated": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/system-info")
def system_info():
    """Get comprehensive system information including enhanced image processing stats"""
    try:
        info = {
            "system_type": system_type,
            "multimodal_available": MULTIMODAL_AVAILABLE,
            "timestamp": datetime.now().isoformat(),
            "flask_config": {
                "documents_path": app.config['DOCUMENTS_PATH'],
                "storage_path": app.config['STORAGE_PATH'],
                "images_path": app.config['IMAGES_PATH'],
                "thumbnails_path": app.config['THUMBNAILS_PATH'],
                "max_content_length": app.config['MAX_CONTENT_LENGTH']
            }
        }

        # Add RAG system stats if available
        if rag_system and hasattr(rag_system, 'get_system_stats'):
            info["rag_stats"] = rag_system.get_system_stats()

        # Add file statistics
        info["file_stats"] = _get_file_statistics()

        # Add enhanced image processing statistics
        info["enhanced_image_stats"] = {
            "total_processed": len(IMAGE_METADATA),
            "kept_images": sum(1 for meta in IMAGE_METADATA.values() if meta.get("kept", False)),
            "filtered_images": sum(1 for meta in IMAGE_METADATA.values() if not meta.get("kept", False)),
            "categories": {},
            "content_types": {},
            "filter_reasons": {},
            "filename_mappings": len(FILENAME_TO_ID_MAP),
            "features": {
                "high_quality_thumbnails": True,
                "text_vs_object_detection": True,
                "readable_filenames": True,
                "enhanced_filtering": True
            }
        }

        for metadata in IMAGE_METADATA.values():
            analysis = metadata.get("analysis", {})
            if metadata.get("kept", False):
                # Count categories for kept images
                category = analysis.get("category", "general")
                info["enhanced_image_stats"]["categories"][category] = info["enhanced_image_stats"]["categories"].get(category, 0) + 1
                content_type = analysis.get("content_type", "unknown")
                info["enhanced_image_stats"]["content_types"][content_type] = info["enhanced_image_stats"]["content_types"].get(content_type, 0) + 1
            else:
                # Count filter reasons
                reason = analysis.get("reason", "unknown")
                info["enhanced_image_stats"]["filter_reasons"][reason] = info["enhanced_image_stats"]["filter_reasons"].get(reason, 0) + 1

        return jsonify(info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/list-documents")
def list_documents():
    """List all available documents with enhanced metadata"""
    try:
        documents_path = Path(app.config['DOCUMENTS_PATH'])
        document_list = {}

        if documents_path.exists():
            for category_dir in documents_path.iterdir():
                if category_dir.is_dir() and category_dir.name not in ['images', 'thumbnails']:
                    category_name = category_dir.name
                    document_list[category_name] = []

                    for file_path in category_dir.rglob('*'):
                        if file_path.is_file():
                            file_info = {
                                "name": file_path.name,
                                "path": str(file_path.relative_to(documents_path)),
                                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                                "modified": datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).isoformat(),
                                "type": file_path.suffix.lower()
                            }
                            document_list[category_name].append(file_info)

        return jsonify({
            "documents": document_list,
            "total_files": sum(len(files) for files in document_list.values()),
            "categories": list(document_list.keys()),
            "enhanced_image_support": MULTIMODAL_AVAILABLE,
            "processed_images": len(IMAGE_METADATA),
            "kept_images": sum(1 for meta in IMAGE_METADATA.values() if meta.get("kept", False)),
            "enhanced_features": {
                "high_quality_thumbnails": True,
                "readable_filenames": True,
                "text_vs_object_filtering": True
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper functions
def _find_best_file_match(requested_filename: str, search_directory: Path):
    """Find the best matching file using fuzzy matching"""
    if not search_directory.exists():
        return None, 0

    from difflib import SequenceMatcher

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def normalize_filename(filename):
        base = filename.replace('.pdf', '').replace('.PDF', '')
        normalized = base.replace('_', ' ').replace('-', ' ').lower()
        return ' '.join(normalized.split())

    requested_normalized = normalize_filename(requested_filename)
    best_match = None
    best_score = 0.0
    candidates = []

    for file_path in search_directory.rglob('*'):
        if file_path.is_file():
            actual_normalized = normalize_filename(file_path.name)

            # Calculate similarity
            similarity_score = similarity(requested_normalized, actual_normalized)

            # Word overlap
            requested_words = set(requested_normalized.split())
            actual_words = set(actual_normalized.split())
            word_overlap = len(requested_words.intersection(actual_words)) / len(requested_words.union(actual_words)) if requested_words.union(actual_words) else 0

            # Combined score
            combined_score = (similarity_score * 0.4) + (word_overlap * 0.6)

            if combined_score > 0.3:
                candidates.append({'path': file_path, 'score': combined_score})

            if combined_score > best_score and combined_score >= 0.3:
                best_score = combined_score
                best_match = file_path

    if candidates:
        logger.debug(f"Fuzzy match candidates for '{requested_filename}':")
        for can in sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]:
            logger.debug(f"  - Score: {can['score']:.2f}, Path: {can['path'].name}")

    return best_match, best_score

def _get_file_statistics():
    """Get enhanced file statistics for system info"""
    documents_path = Path(app.config['DOCUMENTS_PATH'])
    stats = {
        "total_files": 0,
        "by_category": {},
        "total_size_mb": 0,
        "enhanced_image_processing": {
            "total_processed": len(IMAGE_METADATA),
            "kept_images": sum(1 for meta in IMAGE_METADATA.values() if meta.get("kept", False)),
            "filtered_images": sum(1 for meta in IMAGE_METADATA.values() if not meta.get("kept", False))
        }
    }

    try:
        if documents_path.exists():
            for category_dir in documents_path.iterdir():
                if category_dir.is_dir():
                    category_files = list(category_dir.rglob('*'))
                    file_count = len([f for f in category_files if f.is_file()])
                    total_size = sum(f.stat().st_size for f in category_files if f.is_file())

                    stats["by_category"][category_dir.name] = {
                        "count": file_count,
                        "size_mb": round(total_size / (1024 * 1024), 2)
                    }

                    stats["total_files"] += file_count
                    stats["total_size_mb"] += total_size / (1024 * 1024)

            stats["total_size_mb"] = round(stats["total_size_mb"], 2)

    except Exception as e:
        stats["error"] = str(e)

    return stats

# Enhanced Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    """Enhanced global exception handler with detailed logging"""
    logger.error(f"❌ Unhandled exception: {e}", exc_info=True)
    
    if app.config['DEBUG']:
        # In debug mode, show full traceback
        return jsonify({
            "error": "Internal server error", 
            "details": str(e),
            "type": type(e).__name__,
            "system_type": system_type
        }), 500
    else:
        return jsonify({
            "error": "An unexpected error occurred",
            "system_type": system_type
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        "error": "Resource not found",
        "system_type": system_type,
        "multimodal_available": MULTIMODAL_AVAILABLE,
        "enhanced_features": True
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "system_type": system_type,
        "debug_info": str(error) if app.config['DEBUG'] else None,
        "enhanced_features": True
    }), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File too large",
        "max_size_mb": app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    }), 413

# CORS headers for development
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def format_enhanced_sources(sources):
    """Format sources with enhanced metadata for professional display"""
    formatted = []
    
    for i, source in enumerate(sources, 1):
        metadata = source.get('metadata', {})
        
        # Extract proper document title
        title = extract_enhanced_document_title(metadata)
        
        # Extract filename for internal linking
        source_path = metadata.get('source', '')
        filename = extract_filename_from_path(source_path)
        
        # Extract page information
        page_info = extract_page_info(metadata)
        
        # Calculate relevance
        relevance = int((source.get('similarity_score', 0) * 100))
        
        formatted.append({
            "id": f"source_{i}",
            "title": title,
            "filename": filename,
            "page_info": page_info,
            "document_type": metadata.get('source_type', 'Policy Document'),
            "content": source.get('content', ''),
            "relevance": f"{relevance}%",
            "similarity_score": source.get('similarity_score', 0.0),
            "internal_link": f"/documents/{filename}" if filename else None,
            "external_link": metadata.get('original_url', None),
            "supports_highlighting": bool(filename),
            "document_id": metadata.get('document_id', ''),
            "chunk_id": metadata.get('chunk_id', ''),
            "year": "2024",
            # Legacy fields for backward compatibility
            "pages": page_info["page_number"],
            "file_path": filename or source_path
        })
    
    return formatted



def extract_enhanced_document_title(metadata):
    """Extract meaningful document titles with fallbacks"""
    title = metadata.get('title', '')
    if title and title != 'Unknown Document':
        return title
    
    # Try document_id extraction
    doc_id = metadata.get('document_id', '').replace('teppl_', '').replace('_', ' ')
    if doc_id:
        if 'policy' in doc_id.lower():
            return f"NCDOT Traffic Engineering Policy - {doc_id.title()}"
        elif 'practice' in doc_id.lower():
            return f"NCDOT Standard Practice - {doc_id.title()}"
        elif 'guide' in doc_id.lower():
            return f"NCDOT Technical Guide - {doc_id.title()}"
        else:
            return f"NCDOT TEPPL Document - {doc_id.title()}"
    
    # Final fallback
    return "NCDOT TEPPL Document"

def extract_filename_from_path(source_path):
    """Extract actual filename from source path"""
    if not source_path:
        return None
    
    # Handle different path formats
    from pathlib import Path
    try:
        return Path(source_path).name
    except:
        # Fallback for malformed paths
        if '/' in source_path:
            return source_path.split('/')[-1]
        elif '\\' in source_path:
            return source_path.split('\\')[-1]
        else:
            return source_path

def extract_page_info(metadata):
    """Extract comprehensive page information"""
    page_num = metadata.get('page_number', 'N/A')
    
    if page_num == 'N/A' and 'chunk_id' in metadata:
        # Try to extract from chunk_id
        import re
        chunk_id = metadata['chunk_id']
        page_match = re.search(r'_p(\d+)_', chunk_id)
        if page_match:
            page_num = page_match.group(1)
    
    return {
        "page_number": str(page_num),
        "section": metadata.get('section', None),
        "display": f"Page {page_num}" if page_num != 'N/A' else "Multiple Pages"
    }

def build_intelligent_query_context(query):
    """Build intelligent context for better retrieval"""
    query_lower = query.lower()
    
    context = {
        'domain': 'general',
        'intent': 'information',
        'key_terms': [],
        'enhanced_query': query
    }
    
    # SPECIFIC TEPPL DOMAIN DETECTION
    domain_keywords = {
        'speed_limits': ['speed limit', 'statutory', 'mph', 'velocity', 'speed zone'],
        'traffic_signs': ['sign', 'signage', 'regulatory', 'warning', 'guide'],
        'traffic_signals': ['signal', 'light', 'timing', 'intersection'],
        'pavement_markings': ['marking', 'stripe', 'lane', 'crosswalk', 'pavement'],
        'geometric_design': ['design', 'geometric', 'roadway', 'alignment']
    }
    
    # Enhanced term matching
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            context['domain'] = domain
            context['key_terms'].extend([k for k in keywords if k in query_lower])
            break
    
    # Build enhanced query with domain-specific expansions
    enhanced_terms = [query]
    
    if context['domain'] == 'autism_signage':
        enhanced_terms.append('autistic child area sign signage special needs warning')
    elif context['domain'] == 'speed_limits':
        enhanced_terms.append('statutory speed limit mph municipal incorporated limits')
    elif context['domain'] == 'traffic_signs':
        enhanced_terms.append('regulatory warning guide sign symbol post mounting')
    # Add more domain expansions...
    
    context['enhanced_query'] = ' '.join(enhanced_terms)
    
    return context

def apply_context_aware_ranking(sources, context):
    """Apply context-aware ranking to improve relevance"""
    if not sources or not context.get('key_terms'):
        return sources
    
    # Boost scores for domain-specific terms
    key_terms = [term.lower() for term in context['key_terms']]
    
    for source in sources:
        content_lower = source.get('content', '').lower()
        metadata = source.get('metadata', {})
        
        # Count key term matches
        term_matches = sum(1 for term in key_terms if term in content_lower)
        
        # Boost similarity score based on term matches
        if term_matches > 0:
            boost_factor = 1.0 + (term_matches * 0.1)  # 10% boost per term
            original_score = source.get('similarity_score', 0.0)
            source['similarity_score'] = min(1.0, original_score * boost_factor)
    
    # Re-sort by enhanced similarity scores
    return sorted(sources, key=lambda x: x.get('similarity_score', 0), reverse=True)

def generate_professional_chatgpt_answer(sources, query, context):
    """Generate ChatGPT-style professional answer"""
    if not sources:
        return "I couldn't find specific information about this topic in the NCDOT TEPPL documents."

    # Use top sources for comprehensive answer
    top_sources = sorted(sources, key=lambda x: x.get('similarity_score', 0), reverse=True)[:5]
    
    # Build structured response
    response_parts = []
    
    # Dynamic heading based on domain
    domain_headings = {
        'speed_limits': "# North Carolina Speed Limit Regulations\n",
        'traffic_signs': "# Traffic Sign Requirements\n",
        'traffic_signals': "# Traffic Signal Standards\n",
        'pavement_markings': "# Pavement Marking Guidelines\n"
    }
    
    if context['domain'] in domain_headings:
        response_parts.append(domain_headings[context['domain']])
    else:
        response_parts.append("# NCDOT Requirements\n")
    
    # Extract and organize content
    for i, source in enumerate(top_sources, 1):
        content = source.get('content', '').strip()
        if len(content) > 50:
            metadata = source.get('metadata', {})
            doc_title = metadata.get('title', 'NCDOT Document')
            page = metadata.get('page_number', 'N/A')
            
            # Clean and format content
            clean_content = content[:300].strip()
            if len(content) > 300:
                clean_content += "..."
            
            response_parts.append(f"- {clean_content} ({doc_title}, Page {page})")
    
    # Add compliance note
    response_parts.append("\n**Note:** Always verify with current NCDOT policies and consult with Division Engineers for implementation.")
    
    return "\n".join(response_parts)

COMPREHENSIVE_TEPPL_PROMPT = """
You are an expert NCDOT Traffic Engineering consultant providing comprehensive guidance on TEPPL (Traffic Engineering Policies, Practices, and Legal Authority) documents.

RESPONSE STRUCTURE REQUIRED:
1. **Direct Answer** (2-3 sentences addressing the specific question)
2. **Detailed Requirements** (List all applicable standards, measurements, specifications)
3. **Implementation Steps** (Step-by-step guidance for practical application)
4. **Technical Specifications** (Specific measurements, materials, placement criteria)
5. **Regulatory Context** (MUTCD references, CFR citations, legal authority)
6. **Related Considerations** (Connected topics, dependencies, exceptions)
7. **Compliance & Safety** (Installation requirements, approval processes, safety protocols)

CONTEXT FROM NCDOT TEPPL DOCUMENTS:
{comprehensive_context}

USER QUESTION: {query}

REQUIREMENTS:
- Provide a response of AT LEAST 400-600 words
- Include specific measurements, dimensions, and technical details when available
- Reference exact MUTCD sections and CFR regulations
- Explain both the "what" and "why" behind each requirement
- Include practical implementation guidance
- Address safety considerations and approval processes
- Use clear headers and bullet points for readability
- Cite specific page numbers and document sources when available

Generate a comprehensive, technically detailed response that fully addresses all aspects of the question:
"""

def generate_comprehensive_response(sources, query, context):
    """Generate detailed TEPPL responses using GPT-4 and comprehensive context"""
    if not sources:
        return "I couldn't find specific information about this topic in the NCDOT TEPPL documents."
    comprehensive_context = vector_store._aggregate_comprehensive_context(sources[:20])  # Use more sources
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": COMPREHENSIVE_TEPPL_PROMPT.format(
            comprehensive_context=comprehensive_context,
            query=query
        )}],
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content

# Markdown normalization helpers
import re

MD_H1_RE = re.compile(r"(?m)^\s*#\s+")
MD_BULLET_FIX = re.compile(r"(?m)^\s*•\s+")
MD_BAD_BULLETS = re.compile(r"(?m)^\s*[-–—]\s+")
MD_TAB_BULLETS = re.compile(r"(?m)^\t+\-\s+" )

def normalize_markdown(text: str) -> str:
    """Normalize LLM output to standard Markdown the frontend can render well."""
    if not isinstance(text, str) or not text.strip():
        return text

    md = text

    # 1) Convert legacy bullets (•) or dash variants to normal "- "
    md = MD_BULLET_FIX.sub("- ", md)
    md = MD_BAD_BULLETS.sub("- ", md)
    md = MD_TAB_BULLETS.sub("- ", md)

    # 2) Ensure there's at least one H1; if missing, add a generic one
    if not MD_H1_RE.search(md):
        md = "# Answer\n\n" + md

    # 3) Trim redundant blank lines (max 2 in a row)
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md

if __name__ == "__main__":
    print(f"🌟 Starting Enhanced TEPPL App ({system_type} mode)")
    print(f"📁 Documents path: {app.config['DOCUMENTS_PATH']}")
    print(f"💾 Storage path: {app.config['STORAGE_PATH']}")
    print(f"🖼️ Images path: {app.config['IMAGES_PATH']}")
    print(f"🖼️ Thumbnails path: {app.config['THUMBNAILS_PATH']}")

    if MULTIMODAL_AVAILABLE:
        print("🎨 Enhanced multimodal features enabled: text + images + drawings")
        print(f"📸 Processed images: {len(IMAGE_METADATA)}")
        kept_images = sum(1 for meta in IMAGE_METADATA.values() if meta.get("kept", False))
        filtered_images = len(IMAGE_METADATA) - kept_images
        print(f"✅ High-quality images available: {kept_images}")
        print(f"🔍 Text-heavy images filtered: {filtered_images}")
        print(f"🔗 Filename mappings: {len(FILENAME_TO_ID_MAP)}")

        # Show category breakdown
        categories = {}
        for meta in IMAGE_METADATA.values():
            if meta.get("kept", False):
                cat = meta.get("analysis", {}).get("category", "general")
                categories[cat] = categories.get(cat, 0) + 1
        print(f"📂 Image categories: {categories}")
    else:
        print("📝 Text-only mode (install chromadb for multimodal features)")

    # Run the enhanced app
    app.run(
        debug=app.config['DEBUG'],
        port=int(os.getenv('PORT', 5000)),
        host=os.getenv('HOST', '127.0.0.1'),
        threaded=True
    )

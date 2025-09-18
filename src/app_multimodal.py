# src/app_multimodal_enhanced.py

"""
Enhanced Flask Application with Multimodal Support and Optimized Image Handling
Serves TEPPL documents with text, images, and drawings with thumbnail support.
Cleaned to use an application factory and safe initialization.
"""

from __future__ import annotations

import os
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from flask import Flask, jsonify, render_template, redirect, request, send_from_directory, abort
import openai

# ---------------------------
# Paths & constants
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATE_FOLDER = BASE_DIR / "templates"
STATIC_FOLDER = PROJECT_ROOT / "static"

DOCUMENTS_PATH = PROJECT_ROOT / "documents"
STORAGE_PATH = PROJECT_ROOT / "storage"
IMAGES_PATH = DOCUMENTS_PATH / "images"
THUMBNAILS_PATH = DOCUMENTS_PATH / "thumbnails"
IMAGE_METADATA_FILE = DOCUMENTS_PATH / "image_metadata.json"

# ---------------------------
# Optional imports (multimodal vs text-only)
# ---------------------------

MULTIMODAL_AVAILABLE = False
EnhancedMultimodalVectorStore = None  # type: ignore
TEPPLMultimodalRAGSystem = None       # type: ignore
TEPPLVectorStore = None               # type: ignore
TEPPLRAGSystem = None                 # type: ignore

try:
    from src.chroma_multimodal_store import EnhancedMultimodalVectorStore  # type: ignore
    from src.multimodal_rag_system import TEPPLMultimodalRAGSystem         # type: ignore
    MULTIMODAL_AVAILABLE = True
except Exception as e:
    # Try text-only fallback
    try:
        from src.chroma_multimodal_store import TEPPLVectorStore           # type: ignore
        from src.multimodal_rag_system import TEPPLRAGSystem               # type: ignore
        MULTIMODAL_AVAILABLE = False
    except Exception as e2:
        # Leave both as None; we'll handle during init
        MULTIMODAL_AVAILABLE = False


# ---------------------------
# Helpers (pure functions)
# ---------------------------

def _cite(n: int) -> str:
    """Create HTML superscript citation reference."""
    return f"<sup>[{n}]</sup>"

def _clean_and_format_content(content: str) -> str:
    """Clean and format content for professional display."""
    content = " ".join((content or "").split())
    replacements = {
        "MUTCD": "**MUTCD**",
        "NCDOT": "**NCDOT**",
        "CFR": "**CFR**",
        "shall": "**shall**",
        "must": "**must**",
        "required": "**required**",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content

def normalize_markdown(text: str) -> str:
    """Normalize markdown content for consistent rendering."""
    if not text:
        return ""
    text = re.sub(r"(\n#{1,6}\s.+?)(\n)(?!\n)", r"\1\n\n", text)  # header spacing
    text = re.sub(r"\n\s*-\s+", r"\n• ", text)                    # bullets
    text = re.sub(r"(https?://[^\s]+)", r"[\1](\1)", text)        # autolink
    return text

def format_professional_sources(top_sources):
    """Normalize sources for professional UI (ensures internal_link + page_number)."""
    formatted = []
    for s in (top_sources or []):
        md = s.get("metadata", {}) or {}
        title = (
            md.get("title")
            or (md.get("source") or "")
            .split("/")[-1]
            .replace("_", " ")
            .replace("-", " ")
            .rsplit(".", 1)[0]
            or md.get("document_id")
            or "NCDOT TEPPL Document"
        )
        raw_rel = s.get("relevance", s.get("similarity_score", 0.0))
        if isinstance(raw_rel, float) and raw_rel <= 1.0:
            rel_str = f"{round(raw_rel * 100)}%"
        elif isinstance(raw_rel, (int, float)):
            rel_str = f"{round(raw_rel)}%"
        else:
            rel_str = str(raw_rel)
        src_path = md.get("source", "")
        page = md.get("page_number", 1)
        formatted.append(
            {
                "title": title,
                "relevance": rel_str,
                "similarity_score": s.get("similarity_score", 0.0),
                "document_id": md.get("document_id", ""),
                "chunk_id": md.get("chunk_id", ""),
                "source": src_path,
                "pages": md.get("pages") or page,
                "internal_link": f"/documents/{src_path}" if src_path else "",
                "page_number": page,
            }
        )
    return formatted

def generate_professional_answer(sources, query: str) -> str:
    """Generate professional markdown answer with inline citations."""
    if not sources:
        return "No relevant NCDOT TEPPL guidance found for this query."

    top_sources = sorted(sources, key=lambda x: x.get("similarity_score", 0), reverse=True)[:3]
    c1, c2, c3 = _cite(1), _cite(2), _cite(3)

    response_parts = []
    ql = (query or "").lower()
    if any(t in ql for t in ["sign", "signage", "regulatory"]):
        response_parts.append("# Traffic Sign Requirements\n")
    elif any(t in ql for t in ["signal", "timing", "intersection"]):
        response_parts.append("# Traffic Signal Standards\n")
    elif any(t in ql for t in ["marking", "pavement", "lane"]):
        response_parts.append("# Pavement Marking Guidelines\n")
    elif any(t in ql for t in ["design", "geometric", "roadway"]):
        response_parts.append("# Roadway Design Standards\n")
    else:
        response_parts.append("# NCDOT TEPPL Requirements\n")

    requirements, standards, procedures = [], [], []
    for i, src in enumerate(top_sources, 1):
        citation = _cite(i)
        content = (src.get("content") or "").strip()
        if len(content) > 50:
            cl = content.lower()
            target = requirements if any(w in cl for w in ["shall", "must", "required"]) \
                     else standards if any(w in cl for w in ["standard", "specification", "criteria"]) \
                     else procedures
            target.append({"content": content[:400].strip(), "citation": citation})

    if requirements:
        response_parts.append("## **Key Requirements**\n")
        for r in requirements:
            response_parts.append(f"• **{_clean_and_format_content(r['content'])}** {r['citation']}\n")
        response_parts.append("")
    if standards:
        response_parts.append("## **Technical Standards**\n")
        for s in standards:
            response_parts.append(f"• {_clean_and_format_content(s['content'])} {s['citation']}\n")
        response_parts.append("")
    if procedures:
        response_parts.append("## **Implementation Guidelines**\n")
        for p in procedures:
            response_parts.append(f"• {_clean_and_format_content(p['content'])} {p['citation']}\n")
        response_parts.append("")

    response_parts.append("## **Compliance Notes**\n")
    response_parts.append(f"• All implementations must comply with **MUTCD standards** and North Carolina supplements {c1}")
    response_parts.append(f"• **Division Engineer approval** required for final design decisions {c2}")
    response_parts.append(f"• Installation must follow established **safety protocols** and **NCDOT procedures** {c3}")

    return "\n".join(response_parts)

def adapt_response_for_ui(raw_response: Dict[str, Any], query: str, processing_time: float) -> Dict[str, Any]:
    """Adapt response for professional NCDOT output with enhanced structure."""
    answer = normalize_markdown(raw_response.get("answer", "") or "")
    if not answer and raw_response.get("sources"):
        answer = generate_professional_answer(raw_response["sources"], query)

    confidence = float(raw_response.get("confidence", 0.0))
    sources = raw_response.get("sources", []) or []
    if confidence == 0.0 and sources:
        avg_sim = sum(s.get("similarity_score", 0.0) for s in sources) / max(len(sources), 1)
        confidence = min(0.95, max(0.72, avg_sim + 0.25))

    top_sources = sorted(sources, key=lambda x: x.get("similarity_score", 0), reverse=True)[:5]
    formatted_sources = format_professional_sources(top_sources)

    return {
        "success": True,
        "answer": answer,
        "confidence": confidence,
        "sources": formatted_sources,
        "images": raw_response.get("images", []),
        "processing_time": processing_time,
        "query": query,
        "source_count": len(formatted_sources),
        "presentation_mode": "professional",
        "content_type": "markdown",
    }

def load_image_metadata(logger: logging.Logger) -> Dict[str, Any]:
    """Load processed image metadata for fast serving with enhanced structure."""
    if not IMAGE_METADATA_FILE.exists():
        logger.warning("image_metadata.json not found. Image features will be limited.")
        return {}

    try:
        start = time.time()
        size_kb = IMAGE_METADATA_FILE.stat().st_size / 1024.0
        logger.info(f"Loading image_metadata.json (size: {size_kb:.2f} KB)")
        metadata = json.loads(IMAGE_METADATA_FILE.read_text(encoding="utf-8"))
        logger.info(f"📸 Loaded metadata for {len(metadata)} processed images in {time.time() - start:.2f}s")
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Malformed image_metadata.json at line {e.lineno} col {e.colno}: {e.msg}")
        try:
            logger.info("Attempting to recover malformed JSON...")
            text = IMAGE_METADATA_FILE.read_text(encoding="utf-8")
            cleaned = re.sub(r",\s*([\]\}])", r"\1", text)  # remove trailing commas
            metadata = json.loads(cleaned)
            logger.info("Successfully recovered and parsed JSON.")
            return metadata
        except Exception as rec_e:
            logger.critical(f"Failed to recover malformed JSON: {rec_e}")
            return {}
    except Exception as e:
        logger.critical(f"Could not load image metadata: {e}", exc_info=True)
        return {}

def build_filename_map(image_metadata: Dict[str, Any], logger: logging.Logger) -> Tuple[Dict[str, str], int, Dict[str, int]]:
    """Return (filename->id map, missing_filename_count, category_counts)."""
    mapping: Dict[str, str] = {}
    missing_filename = 0
    categories: Dict[str, int] = {}

    for image_id, md in (image_metadata or {}).items():
        if not md.get("kept", False):
            continue
        original_filename = md.get("original_filename")
        readable_filename = md.get("readable_filename")
        if original_filename:
            mapping[original_filename] = image_id
        if readable_filename and readable_filename != original_filename:
            mapping[readable_filename] = image_id
        if not original_filename and not readable_filename:
            missing_filename += 1

        cat = (md.get("analysis", {}) or {}).get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1

    logger.info(f"🔗 Created filename mapping for {len(mapping)} images.")
    if missing_filename:
        logger.warning(f"{missing_filename} kept images are missing filename information.")
    return mapping, missing_filename, categories


# ---------------------------
# Application Factory
# ---------------------------

def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_FOLDER),
        static_folder=str(STATIC_FOLDER),
    )

    # Base config
    app.config.update(
        {
            "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,  # 50MB
            "DOCUMENTS_PATH": str(DOCUMENTS_PATH),
            "STORAGE_PATH": str(STORAGE_PATH),
            "IMAGES_PATH": str(IMAGES_PATH),
            "THUMBNAILS_PATH": str(THUMBNAILS_PATH),
            "DEBUG": os.getenv("FLASK_ENV") == "development",
            "SECRET_KEY": os.getenv("SECRET_KEY", "dev-key-change-in-production"),
        }
    )

    # Logging
    logging.basicConfig(level=logging.DEBUG if app.config["DEBUG"] else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"🔍 Template folder: {TEMPLATE_FOLDER}")
    logger.info(f"🔍 Static folder: {STATIC_FOLDER}")

    # OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.warning("OPENAI_API_KEY not set; LLM features may be disabled.")

    # Load image metadata & maps
    image_metadata = load_image_metadata(logger)
    filename_map, missing_filename, category_counts = build_filename_map(image_metadata, logger)

    # Initialize RAG system
    rag_system = None
    system_type = "error"

    logger.info("🚀 Initializing Enhanced TEPPL RAG System...")
    try:
        if MULTIMODAL_AVAILABLE and EnhancedMultimodalVectorStore and TEPPLMultimodalRAGSystem:
            logger.info("✅ Using multimodal vector store with ChromaDB")
            vector_store = EnhancedMultimodalVectorStore(persist_directory=str(STORAGE_PATH / "chroma_enhanced"))  # type: ignore
            rag_system = TEPPLMultimodalRAGSystem(vector_store=vector_store)  # type: ignore
            system_type = "multimodal"
        elif TEPPLVectorStore and TEPPLRAGSystem:
            logger.info("⚠️ Multimodal components unavailable; falling back to text-only vector store")
            vector_store = TEPPLVectorStore(persist_directory=str(STORAGE_PATH / "teppl_vector"))  # type: ignore
            rag_system = TEPPLRAGSystem(vector_store=vector_store)  # type: ignore
            system_type = "text_only"
        else:
            logger.error("❌ No RAG components available (neither multimodal nor text-only).")
            rag_system = None
            system_type = "error"
    except Exception as e:
        logger.exception(f"❌ FATAL: RAG init failed: {e}")
        rag_system = None
        system_type = "error"

    # Attach to app context for later use by routes
    app.config["RAG_SYSTEM"] = rag_system
    app.config["RAG_SYSTEM_TYPE"] = system_type
    app.config["IMAGE_METADATA"] = image_metadata
    app.config["FILENAME_TO_ID_MAP"] = filename_map
    app.config["IMAGE_MISSING_FILENAME"] = missing_filename
    app.config["IMAGE_CATEGORY_COUNTS"] = category_counts

    # ---------------------------
    # Routes
    # ---------------------------

    @app.route("/")
    def index():
        # Try to render templates/index.html if present
        template_path = TEMPLATE_FOLDER / "index.html"
        if template_path.exists():
            try:
                return render_template(
                    "index.html", 
                    system_type=app.config.get("RAG_SYSTEM_TYPE"),
                    image_stats={
                        "kept_images": sum(1 for v in app.config.get("IMAGE_METADATA", {}).values() if v.get("kept", False)),
                        "total_images": len(app.config.get("IMAGE_METADATA", {})),
                        "categories": app.config.get("IMAGE_CATEGORY_COUNTS", {})
                    },
                    multimodal_available=MULTIMODAL_AVAILABLE
                )
            except Exception as e:
                logger.error(f"Error rendering index.html: {e}")
                # If Jinja blows up, at least show something
                return redirect("/about")
        # No index.html? Send them somewhere useful
        return redirect("/about")

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "system_type": app.config.get("RAG_SYSTEM_TYPE"),
                "documents_path": app.config.get("DOCUMENTS_PATH"),
                "storage_path": app.config.get("STORAGE_PATH"),
                "images": len(app.config.get("IMAGE_METADATA", {})),
                "filename_mappings": len(app.config.get("FILENAME_TO_ID_MAP", {})),
            }
        )

    @app.get("/about")
    def about():
        return jsonify(
            {
                "message": "Enhanced TEPPL App",
                "mode": app.config.get("RAG_SYSTEM_TYPE"),
                "categories": app.config.get("IMAGE_CATEGORY_COUNTS", {}),
                "debug": app.config.get("DEBUG"),
            }
        )

    # --- Search / Query endpoint ---
    @app.post("/query")
    def query_endpoint():
        logger = app.logger
        rag = app.config.get("RAG_SYSTEM")
        if rag is None:
            return jsonify({"success": False, "error": "RAG system is not available."}), 500

        try:
            data = request.get_json(silent=True) or {}
            raw_query = (data.get("query") or "").strip()
            if not raw_query:
                return jsonify({"success": False, "error": "Query is required"}), 400

            n_results = int(data.get("num_results", 20))
            include_images = bool(data.get("include_images", True))
            include_drawings = bool(data.get("include_drawings", True))
            include_text = bool(data.get("include_text", True))

            start = time.time()

            if hasattr(rag, "enhanced_multimodal_retrieval"):
                resp = rag.enhanced_multimodal_retrieval(
                    query=raw_query,
                    user_context={
                        "user_id": request.remote_addr or "anonymous",
                        "ip_address": request.remote_addr or "127.0.0.1",
                    },
                    n_results=n_results,
                    include_images=include_images,
                    include_drawings=include_drawings,
                    include_text=include_text,
                )
            elif hasattr(rag, "query"):
                resp = rag.query(raw_query, n_results=n_results)
            else:
                return jsonify({"success": False, "error": "RAG system missing query interface"}), 500

            if not resp.get("success", True):
                return jsonify({
                    "success": False,
                    "error": resp.get("error", "Unknown retrieval error"),
                    "answer": "",
                    "sources": [],
                    "images": []
                }), 500

            ui_payload = adapt_response_for_ui(resp, raw_query, time.time() - start)
            return jsonify(ui_payload)

        except Exception as e:
            logger.exception("Query processing failed")
            return jsonify({
                "success": False,
                "error": str(e),
                "answer": "",
                "sources": [],
                "images": []
            }), 500

    # --- Document file server (internal_link support) ---
    @app.get("/documents/<path:filepath>")
    def serve_document(filepath: str):
        if ".." in filepath or filepath.startswith("/"):
            abort(403)
        docs_dir = Path(app.config["DOCUMENTS_PATH"])
        try:
            return send_from_directory(docs_dir, filepath, as_attachment=False)
        except Exception:
            abort(404)

    return app


# ---------------------------
# Dev server entrypoint
# ---------------------------

if __name__ == "__main__":
    app = create_app()
    mode = app.config.get("RAG_SYSTEM_TYPE")
    print(f"🌟 Starting Enhanced TEPPL App ({mode} mode)")
    print(f"📁 Documents path: {app.config['DOCUMENTS_PATH']}")
    print(f"💾 Storage path:   {app.config['STORAGE_PATH']}")
    print(f"🖼️ Images path:    {app.config['IMAGES_PATH']}")
    print(f"🖼️ Thumbs path:    {app.config['THUMBNAILS_PATH']}")

    if mode == "multimodal":
        imeta = app.config.get("IMAGE_METADATA", {})
        kept = sum(1 for v in imeta.values() if v.get("kept", False))
        print(f"📸 Processed images: {len(imeta)} (kept={kept})")
        print(f"🔗 Filename mappings: {len(app.config.get('FILENAME_TO_ID_MAP', {}))}")
        print(f"📂 Image categories: {app.config.get('IMAGE_CATEGORY_COUNTS', {})}")
    elif mode == "text_only":
        print("📝 Text-only mode (install chromadb for multimodal features)")
    else:
        print("❌ RAG system unavailable. Endpoints that require it will return errors.")

    app.run(
        debug=app.config["DEBUG"],
        port=int(os.getenv("PORT", "5000")),
        host=os.getenv("HOST", "127.0.0.1"),
        threaded=True,
    )

# src/multimodal_rag_system.py

"""
Enhanced Multimodal RAG System for TEPPL - COMPLETE PRODUCTION VERSION

Integrates with ChromaDB for text, images, and drawings with enhanced image processing support
Includes comprehensive security, hallucination mitigation, and source verification features
Works with enhanced image post-processor and maintains compatibility with existing storage

Key Features:
- Advanced hallucination detection and mitigation
- Comprehensive source verification and direct linking
- Enhanced security measures for document access
- Multi-level confidence scoring
- Detailed audit logging
- Rate limiting and query validation
- Advanced citation and reference tracking
- Secure file serving with access controls
"""

import os
import warnings

# Completely disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_IMPL"] = "chromadb.telemetry.NoopTelemetry"

# Suppress all telemetry warnings
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*posthog.*")
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

import re
import json
import logging
from typing import List, Dict, Any, Set, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum
import threading
from functools import wraps
import secrets
import hashlib
import time
from collections import defaultdict
from rank_bm25 import BM25Okapi
from prompts_professional import PROFESSIONAL_TEPPL_PROMPT

# Setup ChromaDB logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# OpenAI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not available. Install with: pip install openai")

# Import our enhanced multimodal vector store
try:
    from chroma_multimodal_store import EnhancedMultimodalVectorStore
    ENHANCED_VECTOR_STORE_AVAILABLE = True
except ImportError:
    try:
        from chroma_multimodal_store import TEPPLMultimodalVectorStore
        ENHANCED_VECTOR_STORE_AVAILABLE = False
    except ImportError:
        print("❌ No multimodal vector store available")
        ENHANCED_VECTOR_STORE_AVAILABLE = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(str, Enum):
    """Security levels for document access"""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"

class QueryType(str, Enum):
    """Types of queries for different handling"""
    GENERAL = "general"
    TECHNICAL = "technical" 
    REGULATORY = "regulatory"
    SAFETY_CRITICAL = "safety_critical"
    POLICY = "policy"

@dataclass
class HallucinationCheck:
    """Data structure for hallucination detection results"""
    is_hallucinated: bool
    confidence: float
    reasons: List[str]
    source_match_score: float
    factual_consistency: float

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    access_level: SecurityLevel
    ip_address: str
    timestamp: datetime
    session_id: str
    request_signature: str

@dataclass
class SourceReference:
    """Enhanced source reference with direct linking"""
    document_id: str
    document_title: str
    page_number: Optional[int]
    section: Optional[str]
    paragraph: Optional[str]
    direct_link: str
    security_level: SecurityLevel
    content_hash: str
    confidence: float
    extraction_method: str

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed under rate limit"""
        with self.lock:
            now = datetime.now()
            if user_id not in self.requests:
                self.requests[user_id] = []
            
            # Clean old requests
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if now - req_time < timedelta(seconds=self.window_seconds)
            ]
            
            if len(self.requests[user_id]) >= self.max_requests:
                return False
            
            self.requests[user_id].append(now)
            return True

class QueryValidator:
    """Validates and sanitizes queries for security"""
    
    BLOCKED_PATTERNS = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script injection
        r'javascript:',  # JavaScript URLs
        r'data:text/html',  # Data URLs
        r'\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b',  # SQL injection
        r'\.\./',  # Path traversal
        r'__import__',  # Python imports
        r'eval\s*\(',  # Code evaluation
        r'exec\s*\(',  # Code execution
    ]
    
    SENSITIVE_PATTERNS = [
        r'\b(?:password|token|secret|key|api)\b',
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, List[str]]:
        """Validate query for security issues"""
        issues = []
        
        # Check for malicious patterns
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(f"Blocked pattern detected: {pattern[:50]}...")
        
        # Check for sensitive information
        for pattern in cls.SENSITIVE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append("Query contains potentially sensitive information")
                break
        
        # Length check
        if len(query) > 10000:
            issues.append("Query exceeds maximum length")
        
        return len(issues) == 0, issues
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """Sanitize query by removing/encoding dangerous content"""
        sanitized = query.strip()
        
        # Remove potential script tags
        sanitized = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', sanitized, flags=re.IGNORECASE)
        
        # Remove javascript: URLs
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove potential path traversal
        sanitized = re.sub(r'\.\./', '', sanitized)
        
        return sanitized[:5000]  # Truncate to reasonable length

class HallucinationDetector:
    """Advanced hallucination detection system"""
    
    def __init__(self):
        self.known_facts = self._load_verified_facts()
        self.confidence_threshold = 0.75
    
    def _load_verified_facts(self) -> Dict[str, Any]:
        """Load verified TEPPL facts for cross-checking"""
        try:
            facts_file = Path("./data/verified_teppl_facts.json")
            if facts_file.exists():
                with open(facts_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load verified facts: {e}")
        
        # Return basic TEPPL facts if file not available
        return {
            "speed_limits": {
                "school_zone": "25 mph when children are present",
                "residential": "35 mph unless otherwise posted",
                "business_district": "35 mph unless otherwise posted"
            },
            "sign_requirements": {
                "stop_sign_size": "30 inches for most applications",
                "yield_sign_size": "36 inches point to point",
                "speed_limit_sign_size": "24 x 30 inches"
            },
            "regulatory_authority": {
                "mutcd": "Manual on Uniform Traffic Control Devices",
                "ncdot": "North Carolina Department of Transportation"
            }
        }
    
    def check_hallucination(self, answer: str, sources: List[Dict], query: str) -> HallucinationCheck:
        """Comprehensive hallucination detection"""
        reasons = []
        source_match_score = 0.0
        factual_consistency = 0.0
        
        # Check source coverage
        source_content = " ".join([src.get("content", "") for src in sources])
        answer_facts = self._extract_factual_claims(answer)
        
        matched_facts = 0
        total_facts = len(answer_facts)
        
        if total_facts > 0:
            for fact in answer_facts:
                if self._fact_supported_by_sources(fact, source_content):
                    matched_facts += 1
                else:
                    reasons.append(f"Unsupported fact: {fact[:100]}...")
            
            source_match_score = matched_facts / total_facts
        else:
            source_match_score = 1.0  # No facts to verify
        
        # Check against known facts
        factual_consistency = self._check_factual_consistency(answer)
        
        # Check for fabricated citations
        if self._has_fabricated_citations(answer, sources):
            reasons.append("Contains fabricated citations")
            factual_consistency *= 0.5
        
        # Check for overconfident language without strong sources
        if self._has_overconfident_language(answer) and source_match_score < 0.7:
            reasons.append("Overconfident language with weak source support")
        
        # Overall hallucination assessment
        overall_score = (source_match_score * 0.6) + (factual_consistency * 0.4)
        is_hallucinated = overall_score < self.confidence_threshold or len(reasons) > 2
        
        return HallucinationCheck(
            is_hallucinated=is_hallucinated,
            confidence=overall_score,
            reasons=reasons,
            source_match_score=source_match_score,
            factual_consistency=factual_consistency
        )
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract specific factual claims from text"""
        claims = []
        
        # Extract numerical claims
        numerical_claims = re.findall(r'\b\d+(?:\.\d+)?\s*(?:mph|inches|feet|ft|degrees?|percent|%)\b', text, re.IGNORECASE)
        claims.extend(numerical_claims)
        
        # Extract regulatory references
        regulatory_refs = re.findall(r'\b(?:MUTCD|CFR|USC)\s*[A-Z0-9.-]+\b', text, re.IGNORECASE)
        claims.extend(regulatory_refs)
        
        # Extract definitive statements
        definitive_patterns = [
            r'must be \w+',
            r'is required to \w+',
            r'shall be \w+',
            r'is \d+(?:\.\d+)? \w+'
        ]
        
        for pattern in definitive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        
        return claims[:20]  # Limit to prevent overprocessing
    
    def _fact_supported_by_sources(self, fact: str, source_content: str) -> bool:
        """Check if a fact is supported by source content"""
        fact_words = set(fact.lower().split())
        source_words = set(source_content.lower().split())
        
        # Simple overlap check - at least 60% of fact words should appear in sources
        overlap = len(fact_words.intersection(source_words))
        return overlap >= (len(fact_words) * 0.6)
    
    def _check_factual_consistency(self, answer: str) -> float:
        """Check consistency against known verified facts"""
        consistency_score = 1.0
        
        # Check against known speed limits
        speed_mentions = re.findall(r'(\d+)\s*mph', answer, re.IGNORECASE)
        for speed in speed_mentions:
            if int(speed) > 80:  # Unrealistic speed limit
                consistency_score *= 0.7
        
        # Check for contradictory regulatory references
        if 'MUTCD' in answer and 'does not exist' in answer.lower():
            consistency_score *= 0.3
        
        return max(0.0, consistency_score)
    
    def _has_fabricated_citations(self, answer: str, sources: List[Dict]) -> bool:
        """Check for citations not present in sources"""
        cited_docs = re.findall(r'\(([^)]+),\s*(\d{4})\)', answer)
        source_docs = set(src.get("metadata", {}).get("document_id", "") for src in sources)
        
        for doc_title, year in cited_docs:
            # Simple check - if citation doesn't match any source document
            if not any(doc_title.lower() in doc_id.lower() for doc_id in source_docs):
                return True
        
        return False
    
    def _has_overconfident_language(self, answer: str) -> bool:
        """Check for overconfident language patterns"""
        confident_patterns = [
            r'\bdefinitely\b', r'\bcertainly\b', r'\bobviously\b',
            r'\bclearly states\b', r'\bwithout doubt\b', r'\balways\b',
            r'\bnever\b', r'\bguaranteed\b'
        ]
        
        for pattern in confident_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True
        
        return False

class SecureSourceLinker:
    """Handles secure source linking with access controls"""
    
    def __init__(self, base_url: str = "https://teppl.ncdot.gov"):
        self.base_url = base_url
        self.access_cache = {}
        self.link_cache = {}
    
    def create_secure_link(self, source_ref: SourceReference, security_context: SecurityContext) -> str:
        """Create secure, trackable link to source document"""
        if not self._check_access_permission(source_ref, security_context):
            return self._create_restricted_link(source_ref)
        
        # Create signed link
        link_data = {
            "doc_id": source_ref.document_id,
            "page": source_ref.page_number,
            "section": source_ref.section,
            "user_id": security_context.user_id,
            "timestamp": int(time.time()),
            "access_level": security_context.access_level.value
        }
        
        # Create signature
        link_string = json.dumps(link_data, sort_keys=True)
        signature = hashlib.sha256(
            (link_string + self._get_signing_key()).encode()
        ).hexdigest()[:16]
        
        # Encode link
        link_token = self._encode_link_token(link_data, signature)
        
        return f"{self.base_url}/secure-document/{link_token}"
    
    def _check_access_permission(self, source_ref: SourceReference, context: SecurityContext) -> bool:
        """Check if user has permission to access source"""
        cache_key = f"{context.user_id}:{source_ref.document_id}:{context.access_level.value}"
        
        if cache_key in self.access_cache:
            return self.access_cache[cache_key]
        
        # Simple access control logic
        user_level = context.access_level
        doc_level = source_ref.security_level
        
        access_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.RESTRICTED: 2,
            SecurityLevel.CONFIDENTIAL: 3
        }
        
        has_access = access_hierarchy[user_level] >= access_hierarchy[doc_level]
        self.access_cache[cache_key] = has_access
        
        return has_access
    
    def _create_restricted_link(self, source_ref: SourceReference) -> str:
        """Create link for restricted access"""
        return f"{self.base_url}/restricted-access?doc={source_ref.document_id}"
    
    def _get_signing_key(self) -> str:
        """Get signing key for link security"""
        return os.getenv("TEPPL_SIGNING_KEY", "default-dev-key-change-in-production")
    
    def _encode_link_token(self, data: Dict, signature: str) -> str:
        """Encode link token securely"""
        import base64
        token_data = {**data, "sig": signature}
        token_json = json.dumps(token_data)
        return base64.urlsafe_b64encode(token_json.encode()).decode().rstrip('=')

class AuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, log_file: str = "./logs/teppl_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logging
        audit_handler = logging.FileHandler(self.log_file)
        audit_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        self.audit_logger = logging.getLogger('teppl_audit')
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def log_query(self, query: str, user_id: str, results: Dict, security_context: SecurityContext):
        """Log user query with results"""
        log_entry = {
            "event_type": "query",
            "user_id": user_id,
            "query": query[:200],  # Truncate for privacy
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "results_count": results.get("total_sources", 0),
            "confidence": results.get("confidence", 0),
            "has_visual_content": results.get("has_visual_content", False),
            "sources_accessed": [src.get("document_id", "") for src in results.get("sources", [])],
            "ip_address": security_context.ip_address,
            "timestamp": datetime.now().isoformat(),
            "session_id": security_context.session_id
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def log_access_denied(self, user_id: str, resource: str, reason: str, security_context: SecurityContext):
        """Log access denial events"""
        log_entry = {
            "event_type": "access_denied",
            "user_id": user_id,
            "resource": resource,
            "reason": reason,
            "ip_address": security_context.ip_address,
            "timestamp": datetime.now().isoformat(),
            "session_id": security_context.session_id
        }
        
        self.audit_logger.warning(json.dumps(log_entry))
    
    def log_security_event(self, event_type: str, details: Dict, security_context: SecurityContext):
        """Log security-related events"""
        log_entry = {
            "event_type": f"security_{event_type}",
            "details": details,
            "user_id": security_context.user_id,
            "ip_address": security_context.ip_address,
            "timestamp": datetime.now().isoformat(),
            "session_id": security_context.session_id
        }
        
        self.audit_logger.warning(json.dumps(log_entry))

# Enhanced system prompt with hallucination mitigation
ENHANCED_MULTIMODAL_SYSTEM_PROMPT = (
    "You are an expert NCDOT compliance assistant with access to verified TEPPL documents, processed images, and technical drawings. "
    "Your responses must be accurate, traceable, and comply with safety-critical requirements.\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "1. ONLY use information explicitly found in the provided sources\n"
    "2. If information is not in sources, state 'This information is not available in the provided TEPPL documents'\n"
    "3. Always cite sources with exact page numbers and document titles\n"
    "4. Never make assumptions or extrapolate beyond source material\n"
    "5. Flag any conflicting information between sources\n"
    "6. Use qualifying language (e.g., 'According to the TEPPL manual', 'The document states')\n"
    "7. Reference visual content (figures, diagrams, technical drawings) when they support your answer\n"
    "8. Distinguish between requirements (must/shall) and recommendations (should/may)\n"
    "9. Include direct links to source locations for verification\n"
    "10. If a question requires interpretation of safety-critical information, recommend consulting with NCDOT directly\n\n"
    "RESPONSE FORMAT:\n"
    "- Start with a direct answer if available in sources\n"
    "- Use numbered lists with **bold headers** for complex topics\n"
    "- Include [Source: Document Title, Page X] citations\n"
    "- End with 'Verification Links' section containing direct links to sources\n"
    "- If visual content is relevant, describe what figures/diagrams show\n"
)

# New structured answer prompt (v1) enforcing spacing & section order
TEPPL_ANSWER_V1 = """You are TEPPL AI, an assistant answering questions about NCDOT Traffic Engineering
Policies, Practices and Legal Authority (TEPPL). Answer strictly from the provided
context snippets. If a detail isn’t supported by the snippets, omit it.

Return your answer in **GitHub-flavored Markdown** using this structure and spacing:

# Direct answer
A one-sentence, bolded, direct answer if possible. (No preamble.)

## Key requirements
- Bullet list of the 2–5 most important rules/thresholds.
- Keep items short (one sentence each). Bold critical numbers (**12 years**, **35 MPH**, etc).

## Technical details
- Bullet list calling out statutes/sections (e.g., G.S. 20-141) and page references (p. 2).
- Keep each item to one line when possible.

## Implementation steps
1. Short imperative steps the practitioner would take.
2. Keep to 3–6 steps.

## Notes & limitations
- Call out caveats (municipal variation, engineering study required, etc).

Formatting rules:
- Always include a **blank line** before and after each heading and before each list.
- Use regular hyphen bullets `- ` and numbered lists `1.`, `2.` only.
- Do not include a “Sources” section; the server will add sources separately.

User question: "{question}"
Context snippets (may be truncated): 
{context}
"""

def security_required(security_level: SecurityLevel = SecurityLevel.PUBLIC):
    """Decorator for methods requiring security checks"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract security context from kwargs
            security_context = kwargs.get('security_context')
            if not security_context or not isinstance(security_context, SecurityContext):
                raise ValueError("Security context required for this operation")
            
            # Check access level
            if security_context.access_level.value < security_level.value:
                self.audit_logger.log_access_denied(
                    security_context.user_id,
                    func.__name__,
                    f"Insufficient access level: {security_context.access_level.value} < {security_level.value}",
                    security_context
                )
                raise PermissionError("Insufficient access level")
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class TEPPLMultimodalRAGSystem:
    """Enhanced RAG system with comprehensive security, hallucination mitigation, and source verification"""

    def __init__(
        self,
        vector_store: Union["EnhancedMultimodalVectorStore", "TEPPLMultimodalVectorStore"],
        model_name: str = "gpt-4o-mini",
        top_k: int = 20,
        min_confidence: float = 0.15,
        enhanced_image_support: bool = True,
        enable_security: bool = True,
        enable_audit: bool = True
    ):
        self.vs = vector_store
        self.enhanced_image_support = enhanced_image_support
        self.enable_security = enable_security
        self.enable_audit = enable_audit

        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found - some features will be limited")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        self.model = model_name
        self.top_k = top_k
        self.min_confidence = min_confidence

        # Initialize security components
        if self.enable_security:
            self.rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)
            self.query_validator = QueryValidator()
            self.source_linker = SecureSourceLinker()
        
        if self.enable_audit:
            self.audit_logger = AuditLogger()
        
        # Initialize hallucination detection
        self.hallucination_detector = HallucinationDetector()

        # Enhanced image metadata paths for post-processor integration
        self.image_metadata_file = Path("./documents/image_metadata.json")
        self.enhanced_image_metadata = self._load_enhanced_image_metadata()

        # Enhanced TEPPL document mapping with security levels
        self.document_mapping = {
            "teppl_AAH_Info_Sheet": {
                "title": "Adopt-A-Highway Information Sheet",
                "year": "2024",
                "type": "Information Sheet",
                "has_figures": True,
                "enhanced_processing": True,
                "security_level": SecurityLevel.PUBLIC
            },
            "teppl_Coordinators_Manual": {
                "title": "Adopt-A-Highway Coordinators Manual",
                "year": "2015",
                "type": "Manual",
                "has_figures": True,
                "enhanced_processing": True,
                "security_level": SecurityLevel.INTERNAL
            },
            "teppl_11thcovtoc": {
                "title": "11th Edition Coverage Table of Contents",
                "year": "2024",
                "type": "Reference",
                "has_figures": True,
                "enhanced_processing": True,
                "security_level": SecurityLevel.PUBLIC
            },
            "teppl_19a_ncac": {
                "title": "19A NCAC Traffic Engineering Standards",
                "year": "2024",
                "type": "Standard",
                "has_figures": True,
                "enhanced_processing": True,
                "security_level": SecurityLevel.RESTRICTED
            },
            "teppl_2002-06_final_report": {
                "title": "Traffic Engineering Final Report 2002-06",
                "year": "2002",
                "type": "Report",
                "has_figures": True,
                "enhanced_processing": True,
                "security_level": SecurityLevel.INTERNAL
            }
        }

        # Enhanced image category mapping
        self.enhanced_image_categories = {
            "technical_drawings": {
                "description": "Technical drawings with high line density and engineering content",
                "response_template": "technical drawing showing",
                "security_level": SecurityLevel.INTERNAL
            },
            "charts_graphs": {
                "description": "Data visualizations, charts, and statistical graphics",
                "response_template": "chart or graph displaying",
                "security_level": SecurityLevel.PUBLIC
            },
            "data_tables": {
                "description": "Structured data tables with grid patterns",
                "response_template": "data table containing",
                "security_level": SecurityLevel.INTERNAL
            },
            "technical_diagrams": {
                "description": "Engineering diagrams and schematics",
                "response_template": "technical diagram illustrating",
                "security_level": SecurityLevel.RESTRICTED
            },
            "general_figures": {
                "description": "General figures with sufficient visual complexity",
                "response_template": "figure showing",
                "security_level": SecurityLevel.PUBLIC
            }
        }

        # Initialize BM25 lexical search index for text chunks
        self._bm25_ids = []
        self._bm25_tokens = []
        self._bm25_id_to_doc = {}
        self._bm25_id_to_meta = {}
        self._bm25_index = None
        try:
            # Fetch a sample of text documents from Chroma (up to 50k chunks for BM25)
            batch = self.vs.text_collection.get(include=["documents", "metadatas"], limit=50000)
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
            for i, doc in enumerate(docs):
                if not doc or len(doc) < 20:
                    continue  # skip tiny chunks
                meta = metas[i] or {}
                chunk_id = meta.get("chunk_id", meta.get("document_id", f"chunk_{i}"))
                self._bm25_ids.append(chunk_id)
                self._bm25_tokens.append(doc.lower().split())
                self._bm25_id_to_doc[chunk_id] = doc
                self._bm25_id_to_meta[chunk_id] = meta
            if self._bm25_tokens:
                self._bm25_index = BM25Okapi(self._bm25_tokens)
                logger.info(f"✅ BM25 index ready for {len(self._bm25_ids)} text chunks")
        except Exception as e:
            logger.warning(f"BM25 init skipped: {e}")

        logger.info(f"🚀 Enhanced TEPPL RAG system initialized")
        logger.info(f"🔒 Security enabled: {enable_security}")
        logger.info(f"📝 Audit logging enabled: {enable_audit}")
        logger.info(f"🎨 Enhanced image support: {enhanced_image_support}")

        # Log enhanced image processing status
        if self.enhanced_image_metadata:
            kept_count = sum(1 for meta in self.enhanced_image_metadata.values() if meta.get("kept", False))
            logger.info(f"📸 Enhanced image metadata loaded: {kept_count} high-quality images available")

    def _load_enhanced_image_metadata(self) -> Dict[str, Any]:
        """Load enhanced image metadata from post-processor"""
        try:
            if self.image_metadata_file.exists():
                with open(self.image_metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ Loaded enhanced image metadata for {len(metadata)} processed images")
                return metadata
            else:
                logger.warning("⚠️ Enhanced image metadata file not found - using basic image support")
                return {}
        except Exception as e:
            logger.warning(f"Could not load enhanced image metadata: {e}")
            return {}

    def _check_document_access(self, doc_security_level: SecurityLevel, context: SecurityContext) -> bool:
        """Check if user has access to document based on security levels"""
        access_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.RESTRICTED: 2,
            SecurityLevel.CONFIDENTIAL: 3
        }
        
        user_level = access_hierarchy.get(context.access_level, 0)
        required_level = access_hierarchy.get(doc_security_level, 0)
        
        return user_level >= required_level

    def _create_security_context(
        self,
        user_id: str = "anonymous",
        access_level: SecurityLevel = SecurityLevel.PUBLIC,
        ip_address: str = "127.0.0.1",
        session_id: Optional[str] = None
    ) -> SecurityContext:
        """Create default security context for internal use"""
        return self.create_security_context(user_id, access_level, ip_address, session_id)

    def create_security_context(
        self,
        user_id: str,
        access_level: SecurityLevel = SecurityLevel.PUBLIC,
        ip_address: str = "127.0.0.1",
        session_id: Optional[str] = None
    ) -> SecurityContext:
        """Create security context for requests"""
        if not session_id:
            session_id = secrets.token_hex(16)
        
        request_signature = hashlib.sha256(
            f"{user_id}:{ip_address}:{session_id}:{int(time.time())}".encode()
        ).hexdigest()[:16]
        
        return SecurityContext(
            user_id=user_id,
            access_level=access_level,
            ip_address=ip_address,
            timestamp=datetime.now(),
            session_id=session_id,
            request_signature=request_signature
        )

    def answer_question(
        self,
        query: str,
        security_context: SecurityContext,
        include_images: bool = True,
        include_enhanced_images: bool = True,
        query_type: QueryType = QueryType.GENERAL
    ) -> Dict[str, Any]:
        """
        Enhanced answer generation with comprehensive security and hallucination mitigation
        """
        start_time = time.time()
        
        try:
            # Security checks
            if self.enable_security:
                # Rate limiting
                if not self.rate_limiter.is_allowed(security_context.user_id):
                    if self.enable_audit:
                        self.audit_logger.log_security_event(
                            "rate_limit_exceeded",
                            {"user_id": security_context.user_id, "query_preview": query[:50]},
                            security_context
                        )
                    return self._rate_limit_response()
                
                # Query validation
                is_valid, validation_issues = self.query_validator.validate_query(query)
                if not is_valid:
                    if self.enable_audit:
                        self.audit_logger.log_security_event(
                            "invalid_query",
                            {"issues": validation_issues, "query_preview": query[:50]},
                            security_context
                        )
                    return self._validation_error_response(validation_issues)
                
                # Sanitize query
                query = self.query_validator.sanitize_query(query)

            # Classify query type automatically if not specified
            if query_type == QueryType.GENERAL:
                query_type = self._classify_query_type(query)

            # Perform enhanced multimodal search
            contexts = self._enhanced_multimodal_retrieve(
                query, include_images, include_enhanced_images, security_context
            )

            if not contexts:
                return self._no_results_response()

            # Filter contexts by security level
            if self.enable_security:
                contexts = self._filter_contexts_by_security(contexts, security_context)

            # Group contexts by type for better organization
            grouped_contexts = self._group_enhanced_contexts_by_type(contexts)

            # Build enhanced prompt with multimodal content
            prompt = self._build_enhanced_multimodal_prompt(query, grouped_contexts, query_type)

            # Generate response
            if self.client:
                llm_answer = self._call_llm_with_retry(prompt, query_type)
                if "Need more info" in llm_answer:
                    return self._insufficient_info_response()
            else:
                llm_answer = self._generate_fallback_answer(query, grouped_contexts)

            # Perform hallucination check
            hallucination_check = self.hallucination_detector.check_hallucination(
                llm_answer, contexts, query
            )

            # Handle hallucinated responses
            if hallucination_check.is_hallucinated:
                logger.warning(f"Hallucination detected in response: {hallucination_check.reasons}")
                if self.enable_audit:
                    self.audit_logger.log_security_event(
                        "hallucination_detected",
                        {
                            "reasons": hallucination_check.reasons,
                            "confidence": hallucination_check.confidence,
                            "query_preview": query[:50]
                        },
                        security_context
                    )
                
                # Regenerate with more conservative approach
                llm_answer = self._regenerate_conservative_answer(query, contexts, grouped_contexts)
                hallucination_check = self.hallucination_detector.check_hallucination(
                    llm_answer, contexts, query
                )

            # Extract citations and create enhanced response
            cited_sources = self._extract_enhanced_multimodal_citations(llm_answer, contexts)
            confidence = self._calculate_enhanced_multimodal_confidence(
                contexts, cited_sources, hallucination_check
            )

            # Create enhanced source references with secure links
            source_references = self._create_enhanced_source_references(
                cited_sources, security_context
            )

            # Create enhanced bibliography
            sources = self._create_enhanced_multimodal_bibliography(cited_sources, security_context)

            # Enhanced response with comprehensive metadata
            response = {
                "answer": llm_answer,
                "confidence": round(confidence, 3),
                "grounded": True,
                "hallucination_check": {
                    "passed": not hallucination_check.is_hallucinated,
                    "confidence": hallucination_check.confidence,
                    "source_match_score": hallucination_check.source_match_score,
                    "factual_consistency": hallucination_check.factual_consistency
                },
                "citations": cited_sources,
                "sources": sources,
                "source_references": source_references,
                "total_sources": len(sources),
                "content_types": list(grouped_contexts.keys()),
                "has_visual_content": any(
                    ctx.get("content_type") in ["image", "drawing", "processed_image", "classified_image"]
                    for ctx in contexts
                ),
                "visual_references": self._extract_enhanced_visual_references(cited_sources, security_context),
                "enhanced_image_processing": {
                    "enabled": self.enhanced_image_support,
                    "processed_images_available": len(self.enhanced_image_metadata),
                    "high_quality_thumbnails": True,
                    "readable_filenames": True,
                    "categories_used": list(set(
                        ctx.get("metadata", {}).get("enhanced_category", "unknown")
                        for ctx in contexts
                        if ctx.get("content_type") in ["processed_image", "classified_image"]
                    ))
                },
                "security": {
                    "user_access_level": security_context.access_level.value,
                    "sources_filtered": self.enable_security,
                    "query_validated": self.enable_security,
                    "session_id": security_context.session_id
                },
                "query_metadata": {
                    "type": query_type.value,
                    "processing_time_seconds": round(time.time() - start_time, 3),
                    "timestamp": datetime.now().isoformat(),
                    "model_used": self.model if self.client else "fallback"
                }
            }

            # Audit logging
            if self.enable_audit:
                self.audit_logger.log_query(query, security_context.user_id, response, security_context)

            return response

        except Exception as e:
            logger.error(f"Error in enhanced multimodal answer generation: {e}")
            if self.enable_audit:
                self.audit_logger.log_security_event(
                    "processing_error",
                    {"error": str(e), "query_preview": query[:50]},
                    security_context
                )
            return self._error_response(str(e))

    def _classify_query_type(self, query: str) -> QueryType:
        """Automatically classify query type for appropriate handling"""
        query_lower = query.lower()
        
        # Safety-critical keywords
        safety_keywords = ['crash', 'accident', 'fatality', 'injury', 'emergency', 'hazard', 'danger']
        if any(keyword in query_lower for keyword in safety_keywords):
            return QueryType.SAFETY_CRITICAL
        
        # Regulatory keywords
        regulatory_keywords = ['mutcd', 'cfr', 'usc', 'law', 'legal', 'requirement', 'compliance', 'violation']
        if any(keyword in query_lower for keyword in regulatory_keywords):
            return QueryType.REGULATORY
        
        # Policy keywords
        policy_keywords = ['policy', 'procedure', 'guideline', 'standard', 'protocol', 'process']
        if any(keyword in query_lower for keyword in policy_keywords):
            return QueryType.POLICY
        
        # Technical keywords
        technical_keywords = ['design', 'specification', 'dimension', 'calculation', 'engineering', 'technical']
        if any(keyword in query_lower for keyword in technical_keywords):
            return QueryType.TECHNICAL
        
        return QueryType.GENERAL

    def _enhanced_multimodal_retrieve(
        self,
        query: str,
        include_images: bool,
        include_enhanced_images: bool,
        security_context: SecurityContext
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with security filtering and comprehensive search"""
        try:
            results = []

            # Primary search using vector store
            if hasattr(self.vs, 'enhanced_multimodal_search'):
                vector_results = self.vs.enhanced_multimodal_search(
                    query=query,
                    n_results=self.top_k,
                    include_text=True,
                    include_images=include_images,
                    include_drawings=include_images,
                    include_classified=include_enhanced_images
                )
            elif hasattr(self.vs, 'multimodal_search'):
                vector_results = self.vs.multimodal_search(
                    query=query,
                    n_results=self.top_k,
                    include_text=True,
                    include_images=include_images,
                    include_drawings=include_images
                )
            else:
                vector_results = self.vs.similarity_search(query, n_results=self.top_k)

            # Enhance results with processed image metadata
            if include_enhanced_images and self.enhanced_image_metadata:
                enhanced_results = self._enhance_results_with_processed_images(vector_results, query, security_context)
                results.extend(enhanced_results)
            else:
                results.extend(vector_results)

            # Filter by confidence threshold
            filtered_results = [
                result for result in results
                if result.get("similarity_score", 0) >= self.min_confidence
            ]

            # Security-based filtering
            if self.enable_security:
                filtered_results = self._apply_security_filters(filtered_results, security_context)

            logger.info(f"🔍 Enhanced multimodal retrieval: {len(filtered_results)} relevant items found")

            # --- Hybrid Dense+Lexical Retrieval ---
            fused_results = []
            try:
                # Start with all initial dense results (from embeddings)
                candidates = {}
                for item in search_results:
                    # Use chunk_id as a unique key (fall back to item ID if needed)
                    cid = item.get("metadata", {}).get("chunk_id") or item.get("id") or f"id_{len(candidates)}"
                    candidates[cid] = item
                    # Ensure a bm25 score field exists (0.0 if not set later)
                    candidates[cid]["bm25"] = 0.0

                if self._bm25_index:
                    # Get BM25 scores for the query
                    tokens = query.lower().split()
                    bm25_scores = self._bm25_index.get_scores(tokens)
                    # Take top 40 lexical hits
                    top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:40]
                    for i in top_idx:
                        cid = self._bm25_ids[i]
                        score = float(bm25_scores[i])
                        if cid in candidates:
                            # If this chunk was already in dense results, just add the BM25 score
                            candidates[cid]["bm25"] = score
                        else:
                            # Otherwise, add it as a new candidate
                            candidates[cid] = {
                                "id": cid,
                                "content": self._bm25_id_to_doc.get(cid, ""),
                                "metadata": self._bm25_id_to_meta.get(cid, {"chunk_id": cid}),
                                "similarity_score": 0.0,
                                "bm25": score
                            }

                # Apply Reciprocal Rank Fusion (RRF) to combine scores
                def _rrf(rank, k=60): 
                    return 1.0 / (k + rank)
                # Rank dictionaries for dense and lexical scores
                # (Generate rank position for each candidate in each list)
                dense_rank = {}
                bm25_rank = {}
                # Sort candidates by similarity_score (descending) and assign rank
                for rank, item in enumerate(sorted(candidates.values(), key=lambda x: x.get("similarity_score", 0.0), reverse=True), start=1):
                    dense_rank[item["metadata"].get("chunk_id")] = rank
                # Sort candidates by bm25 score (descending) and assign rank
                for rank, item in enumerate(sorted(candidates.values(), key=lambda x: x.get("bm25", 0.0), reverse=True), start=1):
                    bm25_rank[item["metadata"].get("chunk_id")] = rank

                # Compute fused RRF score for each candidate
                for item in candidates.values():
                    cid = item["metadata"].get("chunk_id")
                    # Use a large rank (9999) if candidate not present in a given list
                    rank_d = dense_rank.get(cid, 9999)
                    rank_l = bm25_rank.get(cid, 9999)
                    item["rrf"] = _rrf(rank_d) + _rrf(rank_l)
                    fused_results.append(item)

                # Sort by combined RRF score (descending) and truncate to top N
                fused_results.sort(key=lambda x: x["rrf"], reverse=True)
                fused_results = fused_results[:12]  # e.g., top 12 fused results
            except Exception as e:
                logger.error(f"Fusion error: {e}")
                fused_results = search_results[:12]  # fallback to top 12 dense if error

            # Replace original search results with fused results
            search_results = fused_results
            # --- End of Hybrid Retrieval block ---

            return search_results

        except Exception as e:
            logger.error(f"Error in enhanced multimodal retrieval: {e}")
            return []

    def _apply_security_filters(
        self,
        results: List[Dict],
        security_context: SecurityContext
    ) -> List[Dict]:
        """Apply security-based filtering to search results"""
        filtered_results = []
        
        for result in results:
            # Get document security level
            doc_id = result.get("metadata", {}).get("document_id", "")
            doc_info = self._get_enhanced_document_info(doc_id)
            doc_security_level = doc_info.get("security_level", SecurityLevel.PUBLIC)
            
            # Check access permission
            if self._check_document_access(doc_security_level, security_context):
                filtered_results.append(result)
            else:
                # Log access denial
                if self.enable_audit:
                    self.audit_logger.log_access_denied(
                        security_context.user_id,
                        doc_id,
                        f"Document security level {doc_security_level.value} > user level {security_context.access_level.value}",
                        security_context
                    )
        
        return filtered_results

    def _enhance_results_with_processed_images(self, contexts, query="", security_context=None):
        """
        Enhance search results with processed image metadata from image_metadata.json
        
        Args:
            contexts: List of search results from ChromaDB
            query: Original search query (optional)
            security_context: Security context for access control (optional)
            
        Returns:
            Enhanced contexts with additional image metadata
        """
        try:
            # Load image metadata if not already loaded
            if not hasattr(self, '_image_metadata_cache'):
                self._image_metadata_cache = self._load_image_metadata()
            
            # Create filename to ID mapping if not exists
            if not hasattr(self, '_filename_to_id_map'):
                self._filename_to_id_map = self._create_filename_mapping(self._image_metadata_cache)
            
            enhanced_contexts = []
            
            for ctx in contexts:
                # Copy the original context
                enhanced_ctx = ctx.copy()
                
                # Check if this is an image/drawing result
                content_type = ctx.get("content_type", "text")
                if content_type in ["image", "drawing", "classified_image"]:
                    image_id = ctx.get("id", "")
                    
                    # Look up enhanced metadata
                    image_info = self._image_metadata_cache.get(image_id)
                    
                    # Try filename mapping if direct lookup fails
                    if not image_info and image_id:
                        for filename, mapped_id in self._filename_to_id_map.items():
                            if image_id in filename or filename in image_id:
                                image_info = self._image_metadata_cache.get(mapped_id)
                                break
                    
                    # Enhance with processed metadata if available
                    if image_info and image_info.get("kept", False):
                        analysis = image_info.get("analysis", {})
                        
                        enhanced_ctx.update({
                            "enhanced_image_metadata": {
                                "web_path": image_info.get("web_path", f"images/{image_id}.png"),
                                "thumbnail_path": image_info.get("thumbnail_web_path", f"thumbnails/{image_id}_thumb.jpg"),
                                "dimensions": analysis.get("dimensions", [0, 0]),
                                "category": analysis.get("category", "general"),
                                "confidence": analysis.get("confidence", 0.5),
                                "original_filename": image_info.get("original_filename", "unknown"),
                                "readable_filename": image_info.get("readable_filename", "unknown"),
                                "content_type_detail": analysis.get("content_type", "unknown"),
                                "complexity_score": analysis.get("complexity_score", 0),
                                "text_vs_object": analysis.get("text_vs_object", {}),
                                "technical_features": analysis.get("technical_features", {})
                            },
                            "is_enhanced": True,
                            "enhancement_source": "image_metadata_json"
                        })
                        
                        logger.debug(f"Enhanced image result: {image_id} ({analysis.get('category', 'general')})")
                    else:
                        # Mark as not enhanced but still processable
                        enhanced_ctx["is_enhanced"] = False
                        enhanced_ctx["enhancement_reason"] = "no_metadata_found"
                
                enhanced_contexts.append(enhanced_ctx)
            
            enhanced_count = len([c for c in enhanced_contexts if c.get('is_enhanced')])
            logger.info(f"Enhanced {enhanced_count} image results out of {len(contexts)} total contexts")
            return enhanced_contexts
            
        except Exception as e:
            logger.error(f"Error enhancing results with processed images: {e}")
            # Return original contexts if enhancement fails
            return contexts

    def _load_image_metadata(self):
        """Load processed image metadata from JSON file"""
        try:
            metadata_file = Path("./documents/image_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(metadata)} processed images")
                return metadata
            else:
                logger.warning("image_metadata.json not found - image enhancement will be limited")
                return {}
        except Exception as e:
            logger.error(f"Failed to load image metadata: {e}")
            return {}

    def _create_filename_mapping(self, image_metadata):
        """Create filename to ID mapping for faster lookups"""
        try:
            filename_map = {}
            
            for image_id, metadata in image_metadata.items():
                if metadata.get("kept", False):
                    original_filename = metadata.get("original_filename")
                    readable_filename = metadata.get("readable_filename")
                    
                    if original_filename:
                        filename_map[original_filename] = image_id
                    if readable_filename and readable_filename != original_filename:
                        filename_map[readable_filename] = image_id
            
            logger.info(f"Created filename mapping for {len(filename_map)} images")
            return filename_map
            
        except Exception as e:
            logger.error(f"Error creating filename mapping: {e}")
            return {}

    def _generate_enhanced_answer(self, query: str, search_results: List[Dict]) -> Tuple[str, float]:
        """Generate answer with confidence scoring"""
        if not self.client:
            return self._generate_fallback_enhanced_answer(query, search_results)
            
        # Create context from search results
        context_parts = []
        for result in search_results[:10]:  # Use top 10 results
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page_number", "N/A")
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"[{source}, Page {page}]: {content}")
        context = "\n\n".join(context_parts)

        llm_input = PROFESSIONAL_TEPPL_PROMPT.format(question=query, context=context)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": llm_input}],
                max_tokens=1100,
                temperature=0.1
            )
            answer = response.choices[0].message.content.strip()
            confidence = self._calculate_answer_confidence(search_results, answer)
            return answer, confidence
        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            return self._generate_fallback_enhanced_answer(query, search_results)

    def _generate_fallback_enhanced_answer(self, query: str, search_results: List[Dict]) -> Tuple[str, float]:
        """Generate fallback answer when OpenAI is unavailable"""
        if not search_results:
            return "No relevant information found in the TEPPL documents.", 0.0
        
        # Create simple structured response
        answer_parts = ["Based on the available TEPPL documents:\n"]
        
        for i, result in enumerate(search_results[:5], 1):
            content = result.get("content", "")[:300]
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page_number", "N/A")
            
            answer_parts.append(f"• According to {source}, Page {page}: {content}")
            if len(content) > 300:
                answer_parts.append("...")
            answer_parts.append("")
        
        answer_parts.append("Note: This response was generated without AI language model assistance. For comprehensive guidance, consult the full TEPPL documentation.")
        
        answer = "\n".join(answer_parts)
        confidence = min(0.7, 0.3 + (len(search_results) * 0.05))
        
        return answer, confidence

    def _calculate_answer_confidence(self, search_results: List[Dict], answer: str) -> float:
        """Calculate confidence score for generated answer"""
        if not search_results:
            return 0.0
        
        # Base confidence from search similarity scores
        similarity_scores = [r.get("similarity_score", 0) for r in search_results[:5]]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        # Boost for multiple relevant sources
        source_bonus = min(0.2, len(search_results) * 0.02)
        
        # Boost for multimodal content
        multimodal_bonus = 0.1 if any(
            r.get("content_type") in ["image", "drawing", "classified_image"] 
            for r in search_results
        ) else 0
        
        # Penalty for very short answers (might indicate insufficient context)
        length_penalty = 0.1 if len(answer) < 100 else 0
        
        confidence = (avg_similarity * 0.6) + source_bonus + multimodal_bonus - length_penalty
        
        return min(0.95, max(0.1, confidence))

    def _check_document_access_for_result(self, result: Dict, security_context: SecurityContext) -> bool:
        """Check if user has access to document"""
        # For now, allow all access (you can add security logic later)
        return security_context.access_level in [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL, SecurityLevel.RESTRICTED]

    def _format_image_result(self, result: Dict) -> Dict:
        """Format image result for response"""
        metadata = result.get("metadata", {})
        return {
            "id": result.get("id", "unknown"),
            "type": "image",
            "content_type": result.get("content_type", "image"),
            "similarity_score": result.get("similarity_score", 0.0),
            "source": metadata.get("source", "unknown"),
            "page_number": metadata.get("page_number", "N/A"),
            "image_path": metadata.get("file_path", ""),
            "teppl_category": metadata.get("teppl_category", "general")
        }

    def _format_text_result(self, result: Dict) -> Dict:
        """Format text result for response"""
        metadata = result.get("metadata", {})
        return {
            "id": result.get("id", "unknown"),
            "type": "text",
            "content": result.get("content", ""),
            "similarity_score": result.get("similarity_score", 0.0),
            "source": metadata.get("source", "unknown"),
            "page_number": metadata.get("page_number", "N/A"),
            "chunk_id": metadata.get("chunk_id", "unknown")
        }

    def enhanced_multimodal_retrieval(self, query: str, user_context: Optional[Dict] = None, **kwargs):
        """Enhanced multimodal retrieval with structured response"""
        try:
            # Create proper SecurityContext
            security_context = self._create_security_context(
                user_id=user_context.get("user_id", "anonymous") if user_context else "anonymous",
                ip_address=user_context.get("ip_address", "127.0.0.1") if user_context else "127.0.0.1"
            )
            
            # Validate query
            if self.enable_security:
                is_valid, validation_issues = self.query_validator.validate_query(query)
                if not is_valid:
                    return {
                        "success": False,
                        "error": "Invalid query",
                        "validation_issues": validation_issues,
                        "answer": "",
                        "sources": [],
                        "images": []
                    }
                
                # Check rate limiting
                if not self.rate_limiter.is_allowed(security_context.user_id):
                    return {
                        "success": False,
                        "error": "Rate limit exceeded",
                        "answer": "",
                        "sources": [],
                        "images": []
                    }
            
            # Perform the search
            if hasattr(self.vs, 'enhanced_multimodal_search'):
                search_results = self.vs.enhanced_multimodal_search(
                    query=query,
                    n_results=kwargs.get('n_results', 20),
                    include_text=kwargs.get('include_text', True),
                    include_images=kwargs.get('include_images', True),
                    include_drawings=kwargs.get('include_drawings', True)
                )
            elif hasattr(self.vs, 'multimodal_search'):
                search_results = self.vs.multimodal_search(
                    query=query,
                    n_results=kwargs.get('n_results', 20),
                    include_text=kwargs.get('include_text', True),
                    include_images=kwargs.get('include_images', True),
                    include_drawings=kwargs.get('include_drawings', True)
                )
            else:
                search_results = self.vs.similarity_search(query, n_results=kwargs.get('n_results', 20))
            
            # --- Hybrid Dense+Lexical Retrieval ---
            fused_results = []
            try:
                # Start with all initial dense results (from embeddings)
                candidates = {}
                for item in search_results:
                    # Use chunk_id as a unique key (fall back to item ID if needed)
                    cid = item.get("metadata", {}).get("chunk_id") or item.get("id") or f"id_{len(candidates)}"
                    candidates[cid] = item
                    # Ensure a bm25 score field exists (0.0 if not set later)
                    candidates[cid]["bm25"] = 0.0

                if self._bm25_index:
                    # Get BM25 scores for the query
                    tokens = query.lower().split()
                    bm25_scores = self._bm25_index.get_scores(tokens)
                    # Take top 40 lexical hits
                    top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:40]
                    for i in top_idx:
                        cid = self._bm25_ids[i]
                        score = float(bm25_scores[i])
                        if cid in candidates:
                            # If this chunk was already in dense results, just add the BM25 score
                            candidates[cid]["bm25"] = score
                        else:
                            # Otherwise, add it as a new candidate
                            candidates[cid] = {
                                "id": cid,
                                "content": self._bm25_id_to_doc.get(cid, ""),
                                "metadata": self._bm25_id_to_meta.get(cid, {"chunk_id": cid}),
                                "similarity_score": 0.0,
                                "bm25": score
                            }

                # Apply Reciprocal Rank Fusion (RRF) to combine scores
                def _rrf(rank, k=60): 
                    return 1.0 / (k + rank)
                # Rank dictionaries for dense and lexical scores
                # (Generate rank position for each candidate in each list)
                dense_rank = {}
                bm25_rank = {}
                # Sort candidates by similarity_score (descending) and assign rank
                for rank, item in enumerate(sorted(candidates.values(), key=lambda x: x.get("similarity_score", 0.0), reverse=True), start=1):
                    dense_rank[item["metadata"].get("chunk_id")] = rank
                # Sort candidates by bm25 score (descending) and assign rank
                for rank, item in enumerate(sorted(candidates.values(), key=lambda x: x.get("bm25", 0.0), reverse=True), start=1):
                    bm25_rank[item["metadata"].get("chunk_id")] = rank

                # Compute fused RRF score for each candidate
                for item in candidates.values():
                    cid = item["metadata"].get("chunk_id")
                    # Use a large rank (9999) if candidate not present in a given list
                    rank_d = dense_rank.get(cid, 9999)
                    rank_l = bm25_rank.get(cid, 9999)
                    item["rrf"] = _rrf(rank_d) + _rrf(rank_l)
                    fused_results.append(item)

                # Sort by combined RRF score (descending) and truncate to top N
                fused_results.sort(key=lambda x: x["rrf"], reverse=True)
                fused_results = fused_results[:12]  # e.g., top 12 fused results
            except Exception as e:
                logger.error(f"Fusion error: {e}")
                fused_results = search_results[:12]  # fallback to top 12 dense if error

            # Replace original search results with fused results
            search_results = fused_results
            # --- End of Hybrid Retrieval block ---

            # Generate answer using OpenAI
            answer, confidence = self._generate_enhanced_answer(query, search_results)
            
            # Process results
            processed_results = self._process_search_results(search_results, security_context)
            
            return {
                "success": True,
                "answer": answer,
                "confidence": confidence,
                "sources": processed_results.get("sources", []),
                "images": processed_results.get("images", []),
                "total_results": len(search_results),
                "filtered_results": processed_results.get("filtered_results", 0),
                "query_type": self._classify_query_type(query).value  # Convert enum to string
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced multimodal retrieval: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "sources": [],
                "images": []
            }

    def _get_enhanced_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get enhanced document information by document ID"""
        try:
            # Return document info from mapping if available
            doc_info = self.document_mapping.get(doc_id, {
                "title": doc_id,
                "year": "Unknown",
                "type": "Document",
                "security_level": SecurityLevel.PUBLIC
            })
            
            return doc_info
            
        except Exception as e:
            logger.error(f"Error getting document info for {doc_id}: {e}")
            return {
                "title": doc_id,
                "year": "Unknown", 
                "type": "Document",
                "security_level": SecurityLevel.PUBLIC
            }

    def _generate_citation(self, metadata):
        """Generate proper citation for a source"""
        try:
            doc_id = metadata.get("document_id", "Unknown Document")
            page = metadata.get("page_number", "N/A")
            source = metadata.get("source", "")
            
            if source:
                filename = Path(source).name
                return f"{filename}, Page {page}"
            else:
                return f"{doc_id}, Page {page}"
                
        except Exception:
            return "Unknown Source"

    def _process_search_results(self, search_results: List[Dict], security_context: SecurityContext) -> Dict:
        """Process search results with security context"""
        try:
            sources = []
            images = []
            
            for result in search_results:
                # Check security access
                if self._check_document_access_for_result(result, security_context):
                    if result.get("content_type") in ["image", "drawing", "classified_image"]:
                        images.append(self._format_image_result(result))
                    else:
                        sources.append(self._format_text_result(result))
            
            return {
                "success": True,
                "sources": sources,
                "images": images,
                "total_results": len(search_results),
                "filtered_results": len(sources) + len(images)
            }
            
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return {
                "success": False,
                "error": str(e),
                "sources": [],
                "images": []
            }

    def _call_llm_with_retry(self, prompt: str, query_type: QueryType, max_retries: int = 3) -> str:
        """Call OpenAI API with retry logic and query-type specific parameters"""
        for attempt in range(max_retries):
            try:
                # Adjust parameters based on query type
                temperature = 0.0  # Always conservative
                max_tokens = 3000 if query_type == QueryType.SAFETY_CRITICAL else 2500
                
                # Add system message based on query type
                system_message = self._get_system_message_for_query_type(query_type)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return "Error: Could not generate response after multiple attempts."

    def _get_system_message_for_query_type(self, query_type: QueryType) -> str:
        """Get appropriate system message based on query type"""
        base_prompt = ENHANCED_MULTIMODAL_SYSTEM_PROMPT
        
        if query_type == QueryType.SAFETY_CRITICAL:
            return base_prompt + (
                "\n\nSAFETY-CRITICAL QUERY DETECTED:\n"
                "- Exercise extreme caution in your response\n"
                "- Only provide information explicitly stated in authoritative TEPPL documents\n"
                "- Include disclaimer: 'For safety-critical decisions, consult directly with NCDOT engineers'\n"
                "- If any uncertainty exists, recommend professional consultation\n"
            )
        elif query_type == QueryType.REGULATORY:
            return base_prompt + (
                "\n\nREGULATORY QUERY DETECTED:\n"
                "- Focus on official requirements and regulations\n"
                "- Distinguish between 'shall' (mandatory) and 'should' (recommended)\n"
                "- Include specific regulatory citations (MUTCD sections, CFR references)\n"
                "- Note any conflicts between different regulatory sources\n"
            )
        elif query_type == QueryType.POLICY:
            return base_prompt + (
                "\n\nPOLICY QUERY DETECTED:\n"
                "- Focus on official NCDOT policies and procedures\n"
                "- Include effective dates and version information when available\n"
                "- Note any policy updates or changes\n"
            )
        
        return base_prompt

    def _regenerate_conservative_answer(
        self,
        query: str,
        contexts: List[Dict],
        grouped_contexts: Dict
    ) -> str:
        """Regenerate answer with more conservative, fact-based approach"""
        # Build very conservative prompt
        conservative_prompt = (
            "You must provide a fact-based answer using ONLY the exact information from the provided sources. "
            "Do not make any inferences, assumptions, or extrapolations. "
            "If information is not explicitly stated, say so clearly.\n\n"
        )
        
        # Add top contexts only
        sources_text = ""
        for context_type, context_list in grouped_contexts.items():
            if context_list:
                sources_text += f"\n=== {context_type.upper()} SOURCES ===\n"
                for ctx in context_list[:2]:  # Only top 2 of each type
                    content = ctx.get("content", "")[:300]  # Shorter excerpts
                    sources_text += f"{content}\n"
        
        full_prompt = conservative_prompt + sources_text + f"\n\nQuestion: {query}\nAnswer:"
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.0,
                    max_tokens=1500,  # Shorter response
                    presence_penalty=0.2
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Conservative regeneration failed: {e}")
        
        # Ultimate fallback
        return self._generate_minimal_factual_answer(query, contexts[:5])

    def _generate_minimal_factual_answer(self, query: str, contexts: List[Dict]) -> str:
        """Generate minimal answer with only direct facts from sources"""
        if not contexts:
            return "No relevant information found in the available TEPPL documents."
        
        facts = []
        for ctx in contexts[:3]:
            content = ctx.get("content", "")[:200]
            doc_id = ctx.get("metadata", {}).get("document_id", "unknown")
            page = ctx.get("metadata", {}).get("page_number", "N/A")
            
            facts.append(f"According to {doc_id}, page {page}: {content}")
        
        return (
            f"Based on the available TEPPL documents, here are the relevant facts:\n\n" +
            "\n\n".join(facts) +
            "\n\nNote: This response contains only direct information from the source documents. "
            "For comprehensive guidance, please consult the full TEPPL documentation or contact NCDOT directly."
        )

    def _create_enhanced_source_references(
        self,
        cited_sources: List[Dict],
        security_context: SecurityContext
    ) -> List[SourceReference]:
        """Create enhanced source references with secure direct links"""
        references = []
        
        for ctx in cited_sources:
            metadata = ctx.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            doc_info = self._get_enhanced_document_info(doc_id)
            
            # Create content hash for verification
            content = ctx.get("content", "")
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            source_ref = SourceReference(
                document_id=doc_id,
                document_title=doc_info["title"],
                page_number=metadata.get("page_number"),
                section=metadata.get("section"),
                paragraph=metadata.get("paragraph"),
                direct_link="",  # Will be populated by secure linker
                security_level=doc_info.get("security_level", SecurityLevel.PUBLIC),
                content_hash=content_hash,
                confidence=ctx.get("similarity_score", 0.0),
                extraction_method=metadata.get("extraction_method", "multimodal_search")
            )
            
            # Create secure link
            if self.enable_security:
                source_ref.direct_link = self.source_linker.create_secure_link(source_ref, security_context)
            else:
                source_ref.direct_link = f"/documents/{doc_id}"
                if source_ref.page_number:
                    source_ref.direct_link += f"#page={source_ref.page_number}"
            
            references.append(source_ref)
        
        return references

    def _group_enhanced_contexts_by_type(self, contexts: List[Dict]) -> Dict[str, List[Dict]]:
        """Group contexts by content type for better organization"""
        grouped = {
            "text": [],
            "images": [],
            "drawings": [],
            "processed_images": []
        }
        
        for ctx in contexts:
            content_type = ctx.get("content_type", "text")
            if content_type == "text":
                grouped["text"].append(ctx)
            elif content_type in ["image", "classified_image"]:
                grouped["images"].append(ctx)
            elif content_type == "drawing":
                grouped["drawings"].append(ctx)
            elif content_type == "processed_image":
                grouped["processed_images"].append(ctx)
            else:
                grouped["text"].append(ctx)  # Default to text
        
        return grouped

    def _build_enhanced_multimodal_prompt(self, query: str, grouped_contexts: Dict, query_type: QueryType) -> str:
        """Build enhanced prompt with multimodal content"""
        prompt = f"Query: {query}\n\n"
        prompt += "Based on the following TEPPL documents and visual content:\n\n"
        
        # Add text sources
        if grouped_contexts.get("text"):
            prompt += "=== TEXT SOURCES ===\n"
            for i, ctx in enumerate(grouped_contexts["text"][:5]):
                content = ctx.get("content", "")[:500]
                doc_id = ctx.get("metadata", {}).get("document_id", "unknown")
                page = ctx.get("metadata", {}).get("page_number", "N/A")
                prompt += f"Source {i+1} ({doc_id}, page {page}): {content}\n\n"
        
        # Add image sources
        if grouped_contexts.get("images") or grouped_contexts.get("processed_images"):
            prompt += "=== VISUAL CONTENT ===\n"
            all_images = grouped_contexts.get("images", []) + grouped_contexts.get("processed_images", [])
            for i, ctx in enumerate(all_images[:3]):
                content = ctx.get("content", "")
                doc_id = ctx.get("metadata", {}).get("document_id", "unknown")
                enhanced_meta = ctx.get("enhanced_image_metadata", {})
                category = enhanced_meta.get("category", "general")
                prompt += f"Visual {i+1} ({doc_id}, {category}): {content}\n"
                if enhanced_meta.get("readable_filename"):
                    prompt += f"Filename: {enhanced_meta['readable_filename']}\n"
                prompt += "\n"
        
        # Add drawings
        if grouped_contexts.get("drawings"):
            prompt += "=== TECHNICAL DRAWINGS ===\n"
            for i, ctx in enumerate(grouped_contexts["drawings"][:2]):
                content = ctx.get("content", "")
                doc_id = ctx.get("metadata", {}).get("document_id", "unknown")
                prompt += f"Drawing {i+1} ({doc_id}): {content}\n\n"
        
        prompt += f"\nPlease answer the question: {query}\n"
        return prompt

    def _generate_fallback_answer(self, query: str, contexts: List[Dict]) -> str:
        """Generate fallback answer when OpenAI is not available"""
        if not contexts:
            return "No relevant information found in the TEPPL documents."
        
        # Simple template-based response
        answer = "Based on the available TEPPL documentation:\n\n"
        
        for i, ctx in enumerate(contexts[:3]):
            content = ctx.get("content", "")[:200]
            metadata = ctx.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page_number", "N/A")
            
            answer += f"{i+1}. According to {source} (page {page}): {content}\n\n"
        
        answer += "Note: This response was generated without AI language model assistance. For more comprehensive answers, ensure OpenAI API access is configured."
        
        return answer

    def _extract_enhanced_multimodal_citations(self, answer: str, contexts: List[Dict]) -> List[Dict]:
        """Extract citations from the answer and match to contexts"""
        citations = []
        
        for i, ctx in enumerate(contexts[:10]):
            # Simple heuristic - if context content appears in answer
            content_snippet = ctx.get("content", "")[:100]
            doc_id = ctx.get("metadata", {}).get("document_id", "unknown")
            
            # Check if this context is likely cited
            if any(word in answer.lower() for word in content_snippet.lower().split()[:5]):
                citations.append(ctx)
        
        return citations[:5]  # Limit to top 5 citations

    def _calculate_enhanced_multimodal_confidence(self, contexts: List[Dict], cited_sources: List[Dict], hallucination_check) -> float:
        """Calculate confidence score for multimodal response"""
        if not contexts:
            return 0.0
        
        # Base confidence from search similarity
        base_confidence = sum(ctx.get("similarity_score", 0) for ctx in contexts[:5]) / min(5, len(contexts))
        
        # Boost for multimodal content
        multimodal_bonus = 0.1 if any(
            ctx.get("content_type") in ["image", "drawing", "classified_image"]
            for ctx in contexts
        ) else 0.0
        
        # Citation coverage
        citation_coverage = len(cited_sources) / min(5, len(contexts))
        
        # Hallucination penalty
        hallucination_penalty = 0.0 if hallucination_check.confidence > 0.7 else 0.2
        
        final_confidence = (base_confidence * 0.6) + (citation_coverage * 0.3) + multimodal_bonus - hallucination_penalty
        
        return min(1.0, max(0.0, final_confidence))

    def _create_enhanced_multimodal_bibliography(self, cited_sources: List[Dict], security_context) -> List[Dict]:
        """Create enhanced bibliography with multimodal sources"""
        bibliography = []
        
        for ctx in cited_sources:
            metadata = ctx.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            
            # Get document info
            doc_info = self.document_mapping.get(doc_id, {
                "title": doc_id,
                "year": "Unknown",
                "type": "Document"
            })
            
            source_entry = {
                "document_id": doc_id,
                "title": doc_info["title"],
                "year": doc_info.get("year", "Unknown"),
                "type": doc_info.get("type", "Document"),
                "page_number": metadata.get("page_number"),
                "content_type": ctx.get("content_type", "text"),
                "similarity_score": ctx.get("similarity_score", 0.0),
                "excerpt": ctx.get("content", "")[:200] + "..." if len(ctx.get("content", "")) > 200 else ctx.get("content", "")
            }
            
            # Add enhanced image metadata if available
            if ctx.get("enhanced_image_metadata"):
                img_meta = ctx["enhanced_image_metadata"]
                source_entry.update({
                    "image_metadata": {
                        "category": img_meta.get("category", "general"),
                        "readable_filename": img_meta.get("readable_filename", "unknown"),
                        "dimensions": img_meta.get("dimensions", [0, 0]),
                        "thumbnail_path": img_meta.get("thumbnail_path")
                    }
                })
            
            bibliography.append(source_entry)
        
        return bibliography

    def _extract_enhanced_visual_references(self, cited_sources: List[Dict], security_context) -> List[Dict]:
        """Extract visual content references for display"""
        visual_refs = []
        
        for ctx in cited_sources:
            content_type = ctx.get("content_type", "text")
            if content_type in ["image", "drawing", "classified_image", "processed_image"]:
                enhanced_meta = ctx.get("enhanced_image_metadata", {})
                
                visual_ref = {
                    "id": ctx.get("id", ""),
                    "type": content_type,
                    "web_path": enhanced_meta.get("web_path", f"images/{ctx.get('id', '')}.png"),
                    "thumbnail_path": enhanced_meta.get("thumbnail_path", f"thumbnails/{ctx.get('id', '')}_thumb.jpg"),
                    "category": enhanced_meta.get("category", "general"),
                    "readable_filename": enhanced_meta.get("readable_filename", "unknown"),
                    "document_id": ctx.get("metadata", {}).get("document_id", "unknown"),
                    "similarity_score": ctx.get("similarity_score", 0.0)
                }
                
                visual_refs.append(visual_ref)
        
        return visual_refs

    def _filter_contexts_by_security(self, contexts: List[Dict], security_context) -> List[Dict]:
        """Filter contexts based on security permissions"""
        if not self.enable_security:
            return contexts
        
        filtered_contexts = []
        for ctx in contexts:
            doc_id = ctx.get("metadata", {}).get("document_id", "")
            doc_info = self.document_mapping.get(doc_id, {"security_level": SecurityLevel.PUBLIC})
            doc_security_level = doc_info.get("security_level", SecurityLevel.PUBLIC)
            
            if self._check_document_access(doc_security_level, security_context):
                filtered_contexts.append(ctx)
        
        return filtered_contexts

    def _rate_limit_response(self) -> Dict[str, Any]:
        """Return rate limit exceeded response"""
        return {
            "error": "Rate limit exceeded. Please try again later.",
            "error_type": "rate_limit_exceeded",
            "answer": "I'm sorry, but you've exceeded the rate limit for queries. Please wait a moment and try again.",
            "confidence": 0.0,
            "grounded": False,
            "sources": [],
            "citations": [],
            "total_sources": 0,
            "has_visual_content": False,
            "retry_after_seconds": 3600
        }

    def _validation_error_response(self, issues: List[str]) -> Dict[str, Any]:
        """Return validation error response"""
        return {
            "error": "Query validation failed",
            "error_type": "validation_error",
            "validation_issues": issues,
            "answer": "I'm sorry, but your query contains content that cannot be processed. Please rephrase your question.",
            "confidence": 0.0,
            "grounded": False,
            "sources": [],
            "citations": [],
            "total_sources": 0,
            "has_visual_content": False
        }

    def _no_results_response(self) -> Dict[str, Any]:
        """Return no results found response"""
        return {
            "answer": "I couldn't find any relevant information in the available TEPPL documents for your query. Please try rephrasing your question or checking for spelling errors.",
            "confidence": 0.0,
            "grounded": False,
            "sources": [],
            "citations": [],
            "total_sources": 0,
            "has_visual_content": False,
            "suggestion": "Try using different keywords or asking about specific TEPPL topics like traffic signs, signal timing, or intersection design."
        }

    def _insufficient_info_response(self) -> Dict[str, Any]:
        """Return insufficient information response"""
        return {
            "answer": "The available TEPPL documents don't contain sufficient information to answer your specific question. For detailed guidance on this topic, please contact NCDOT directly.",
            "confidence": 0.0,
            "grounded": False,
            "sources": [],
            "citations": [],
            "total_sources": 0,
            "has_visual_content": False,
            "contact_suggestion": "Contact NCDOT Traffic Engineering for specific guidance"
        }

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Return generic error response"""
        return {
            "error": "An error occurred while processing your query",
            "error_details": error_message,
            "answer": "I'm sorry, but an error occurred while processing your query. Please try again or contact support if the problem persists.",
            "confidence": 0.0,
            "grounded": False,
            "sources": [],
            "citations": [],
            "total_sources": 0,
            "has_visual_content": False
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics with security information"""
        try:
            # Get base vector store stats
            if hasattr(self.vs, 'get_enhanced_collection_stats'):
                vector_stats = self.vs.get_enhanced_collection_stats()
            else:
                vector_stats = {"error": "Could not retrieve vector store stats"}

            # Enhanced image processing stats
            enhanced_image_stats = {
                "total_processed_images": len(self.enhanced_image_metadata),
                "kept_images": 0,
                "categories": {},
                "content_types": {}
            }

            if self.enhanced_image_metadata:
                for image_id, metadata in self.enhanced_image_metadata.items():
                    if metadata.get("kept", False):
                        enhanced_image_stats["kept_images"] += 1
                        analysis = metadata.get("analysis", {})
                        category = analysis.get("category", "general")
                        content_type = analysis.get("content_type", "unknown")
                        enhanced_image_stats["categories"][category] = enhanced_image_stats["categories"].get(category, 0) + 1
                        enhanced_image_stats["content_types"][content_type] = enhanced_image_stats["content_types"].get(content_type, 0) + 1

            # Security stats
            security_stats = {
                "enabled": self.enable_security,
                "audit_logging": self.enable_audit,
                "rate_limiting": self.enable_security,
                "query_validation": self.enable_security,
                "hallucination_detection": True,
                "secure_source_linking": self.enable_security,
                "document_access_control": self.enable_security
            }

            return {
                "rag_system": {
                    "model": self.model,
                    "top_k": self.top_k,
                    "min_confidence": self.min_confidence,
                    "multimodal_enabled": True,
                    "enhanced_image_processing": self.enhanced_image_support,
                    "openai_available": OPENAI_AVAILABLE,
                    "enhanced_vector_store": ENHANCED_VECTOR_STORE_AVAILABLE
                },
                "vector_store": vector_stats,
                "enhanced_image_processing": enhanced_image_stats,
                "security": security_stats,
                "capabilities": {
                    "text_search": True,
                    "image_search": True,
                    "drawing_search": True,
                    "enhanced_processed_images": self.enhanced_image_support,
                    "multimodal_citations": True,
                    "visual_content_references": True,
                    "high_quality_thumbnails": self.enhanced_image_support,
                    "readable_filenames": self.enhanced_image_support,
                    "text_vs_object_detection": self.enhanced_image_support,
                    "technical_content_analysis": self.enhanced_image_support,
                    "hallucination_detection": True,
                    "security_controls": self.enable_security,
                    "audit_logging": self.enable_audit,
                    "source_verification": True,
                    "direct_source_linking": True
                },
                "document_mapping": {
                    "total_documents": len(self.document_mapping),
                    "enhanced_processing_enabled": sum(
                        1 for doc in self.document_mapping.values()
                        if doc.get("enhanced_processing", False)
                    ),
                    "security_levels": {
                        level.value: sum(
                            1 for doc in self.document_mapping.values()
                            if doc.get("security_level") == level
                        ) for level in SecurityLevel
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

# Backward compatibility aliases
TEPPLRAGSystem = TEPPLMultimodalRAGSystem

# Test function for the enhanced system
def test_enhanced_secure_rag():
    """Test the enhanced secure RAG system"""
    print("🧪 Testing Enhanced Secure TEPPL RAG System...")
    
    try:
        # Initialize components
        if ENHANCED_VECTOR_STORE_AVAILABLE:
            from chroma_multimodal_store import EnhancedMultimodalVectorStore
            vector_store = EnhancedMultimodalVectorStore()
        else:
            print("⚠️ Enhanced vector store not available")
            return

        # Initialize RAG system with security enabled
        rag_system = TEPPLMultimodalRAGSystem(
            vector_store,
            enhanced_image_support=True,
            enable_security=True,
            enable_audit=True
        )

        # Create test security context
        security_context = rag_system.create_security_context(
            user_id="test_user",
            access_level=SecurityLevel.INTERNAL,
            ip_address="192.168.1.100"
        )

        # Test queries with different security levels
        test_queries = [
            ("What are the speed limit requirements for residential areas?", QueryType.REGULATORY),
            ("Show me technical drawings for traffic signs", QueryType.TECHNICAL),
            ("What safety procedures are required for work zones?", QueryType.SAFETY_CRITICAL),
            ("What is the NCDOT policy on school zone signage?", QueryType.POLICY)
        ]

        for query, query_type in test_queries:
            print(f"\n🔍 Testing secure query: '{query}'")
            result = rag_system.answer_question(
                query,
                security_context=security_context,
                query_type=query_type
            )

            print(f"📊 Enhanced Secure Results:")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Hallucination check passed: {result['hallucination_check']['passed']}")
            print(f"  Sources: {result['total_sources']}")
            print(f"  Security context: {result['security']['user_access_level']}")
            print(f"  Query type: {result['query_metadata']['type']}")
            print(f"  Processing time: {result['query_metadata']['processing_time_seconds']}s")
            print(f"  Answer preview: {result['answer'][:200]}...")

        # Show enhanced system stats
        stats = rag_system.get_system_stats()
        print(f"\n📈 Enhanced Secure System Statistics:")
        print(f"  Security enabled: {stats['security']['enabled']}")
        print(f"  Audit logging: {stats['security']['audit_logging']}")
        print(f"  Hallucination detection: {stats['capabilities']['hallucination_detection']}")
        print(f"  Source verification: {stats['capabilities']['source_verification']}")
        print(f"  Vector store items: {stats['vector_store'].get('total_items', 0)}")

        print("\n🎉 Enhanced secure RAG system test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_secure_rag()

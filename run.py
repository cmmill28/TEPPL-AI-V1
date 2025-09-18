#!/usr/bin/env python3
"""TEPPL AI Flask App Launcher"""

import os
import sys
from pathlib import Path

# Load environment variables FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import the app factory from the enhanced module
from app_multimodal import create_app

app = create_app()  # build the Flask app instance

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API Key loaded (ends with: ...{api_key[-4:]})")
    else:
        print("❌ WARNING: OPENAI_API_KEY not found in environment variables")

    print("🚀 Starting TEPPL AI Server...")
    print(f"📍 Host: {os.environ.get('HOST', '127.0.0.1')}")
    print(f"🔌 Port: {os.environ.get('PORT', '5000')}")
    print("🔄 Auto-reload: DISABLED")
    print("🧵 Threading: ENABLED")
    print(f"🎛️ Mode: {app.config.get('RAG_SYSTEM_TYPE')}")

    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
        threaded=True,
    )

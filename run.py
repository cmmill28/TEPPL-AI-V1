#!/usr/bin/env python3
"""TEPPL AI Flask App Launcher"""

import os
import sys
from pathlib import Path

# Load environment variables FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads your .env file
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from app_multimodal import app

if __name__ == "__main__":
    # Verify OpenAI API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API Key loaded (ends with: ...{api_key[-4:]})")
    else:
        print("❌ WARNING: OPENAI_API_KEY not found in environment variables")
    
    print(f"🚀 Starting TEPPL AI Server...")
    print(f"📍 Host: {os.environ.get('HOST', '127.0.0.1')}")
    print(f"🔌 Port: {os.environ.get('PORT', '5000')}")
    print(f"🔄 Auto-reload: DISABLED")
    print(f"🧵 Threading: ENABLED")
    
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
        threaded=True
    )

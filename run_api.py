#!/usr/bin/env python3
"""
Startup script for the Shopping Agent API
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["GEMINI_API_KEY", "ZILLIZ_TOKEN", "ZILLIZ_URI"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        sys.exit(1)
    
    print("Starting Shopping Agent API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Health check available at: http://localhost:8000/health")
    
    try:
        uvicorn.run(
            "app.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

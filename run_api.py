#!/usr/bin/env python3
"""
startup script for the shopping agent api
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# add the project root to python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    # load environment variables from .env file
    load_dotenv()
    
    # check for required environment variables
    required_vars = ["GEMINI_API_KEY", "ZILLIZ_TOKEN", "ZILLIZ_URI"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nplease set these variables in your .env file or environment.")
        sys.exit(1)
    
    print("starting shopping agent api...")
    print("api documentation will be available at: http://localhost:8000/docs")
    print("health check available at: http://localhost:8000/health")
    
    try:
        uvicorn.run(
            "app.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nshutting down api server...")
    except Exception as e:
        print(f"failed to start server: {e}")
        sys.exit(1)

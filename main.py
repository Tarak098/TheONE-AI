"""
TheONE AI - Application Entry Point for Production (Render, Docker, Local).

Exports the ASGI FastAPI application instance `app` from `backend.main`.
Allows both standard start commands:
    uvicorn main:app --host 0.0.0.0 --port $PORT
and
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT
"""

import os
import uvicorn
from backend.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port)


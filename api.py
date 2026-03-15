"""
API FastAPI independiente — Deteccion de zonas de revalorizacion.
Puerto: 8090
Arranque: uvicorn api:app --host 127.0.0.1 --port 8090
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ML Revalorizacion — Zonas de Madrid",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://adrianmoreno-dev.com", "http://127.0.0.1", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from router import router
app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "proyecto": "deteccion-zonas-revalorizacion"}

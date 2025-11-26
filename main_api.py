"""
EffiCode Analyzer API - Main Application Entry Point

Este es el punto de entrada principal de la API. 
Toda la lógica de negocio está modularizada en la carpeta api/.

Estructura:
    api/
    ├── deps.py          # Dependencias compartidas (ServiceContainer)
    ├── schemas/         # Modelos Pydantic
    │   └── analysis.py  # Schemas para análisis
    └── routers/         # Endpoints organizados por dominio
        └── analysis.py  # Router de análisis de complejidad
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Asegurar que el directorio actual esté en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Importar módulos de la API
from api.deps import ServiceContainer
from api.routers import analysis_router

load_dotenv()


# Lifespan context manager (reemplaza @app.on_event deprecated)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Inicializa servicios al arrancar y los limpia al cerrar.
    """
    # Startup
    print("🚀 Starting EffiCode Analyzer API...")
    ServiceContainer.initialize()
    print("✅ All services initialized successfully")
    
    yield  # La aplicación se ejecuta aquí
    
    # Shutdown
    print("👋 Shutting down EffiCode Analyzer API...")


# Crear aplicación FastAPI
app = FastAPI(
    title="EffiCode Analyzer API",
    description="API para análisis de complejidad algorítmica de pseudocódigo estilo Cormen",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Incluir routers
# El prefijo /analysis se define en el router, así que /analyze queda como /analysis/analyze
# Para mantener compatibilidad con el frontend, montamos sin prefijo adicional
app.include_router(analysis_router, prefix="", tags=["Analysis"])


# Endpoint raíz
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "name": "EffiCode Analyzer API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/analysis/health"
    }


if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)

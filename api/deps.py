"""
Dependencias compartidas para toda la API.
Aquí se inicializan y gestionan los servicios que son compartidos entre routers.
"""
from typing import Dict, Any, Optional

from Servicios.Grammar import Grammar
from Servicios.LLMService import LLMService
from Modelos.Parser import Parser
from Modelos.Analizador import Analizador


class ServiceContainer:
    """
    Contenedor de servicios singleton.
    Gestiona la inicialización y acceso a los servicios compartidos.
    """
    _instance: Optional['ServiceContainer'] = None
    _services: Dict[str, Any] = {}
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls) -> None:
        """Inicializa todos los servicios necesarios."""
        if cls._initialized:
            print("⚠️  Services already initialized.")
            return
        
        try:
            print("--- Initializing Services ---")
            
            # Servicios base
            cls._services["grammar"] = Grammar()
            cls._services["llm_service"] = LLMService()
            
            # Servicios que dependen de otros
            cls._services["parser"] = Parser(
                id=1, 
                gramatica=cls._services["grammar"], 
                llm_service=cls._services["llm_service"]
            )
            cls._services["analizador"] = Analizador(
                id=1, 
                parser=cls._services["parser"], 
                llm_service=cls._services["llm_service"]
            )
            
            cls._initialized = True
            print("✅ Services initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing services: {e}")
            raise
    
    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """Obtiene la instancia singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def grammar(self) -> Grammar:
        return self._services.get("grammar")
    
    @property
    def llm_service(self) -> LLMService:
        return self._services.get("llm_service")
    
    @property
    def parser(self) -> Parser:
        return self._services.get("parser")
    
    @property
    def analizador(self) -> Analizador:
        return self._services.get("analizador")
    
    @classmethod
    def is_ready(cls) -> bool:
        """Verifica si todos los servicios están inicializados."""
        required = ["grammar", "llm_service", "parser", "analizador"]
        return all(key in cls._services for key in required)


def get_services() -> ServiceContainer:
    """
    Dependency injection para obtener los servicios.
    Uso: services = Depends(get_services)
    """
    if not ServiceContainer.is_ready():
        raise RuntimeError("Services not initialized. Call ServiceContainer.initialize() first.")
    return ServiceContainer.get_instance()

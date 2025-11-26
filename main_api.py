from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import base64
import pydot
from io import BytesIO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from Servicios.Grammar import Grammar
from Servicios.LLMService import LLMService
from Modelos.Parser import Parser
from Modelos.Analizador import Analizador
from Modelos.Algoritmo import Algoritmo
from Enumerations.tipoAlgoritmo import TipoAlgoritmo
from Modelos.Reporte import Reporte

load_dotenv()

app = FastAPI(title="EffiCode Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

services = {}

@app.on_event("startup")
async def startup_event():
    try:
        print("--- Initializing Services ---")
        services["grammar"] = Grammar()
        services["llm_service"] = LLMService()
        services["parser"] = Parser(id=1, gramatica=services["grammar"], llm_service=services["llm_service"])
        services["analizador"] = Analizador(id=1, parser=services["parser"], llm_service=services["llm_service"])
        print("✅ Services initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        pass

class AnalysisRequest(BaseModel):
    pseudocode: str

class AnalysisResponse(BaseModel):
    complexity_o: str
    complexity_omega: str
    complexity_theta: str
    justification: str
    justification_data: dict
    validation: str
    ast_image: str  # Base64 encoded PNG

def generate_ast_image_base64(ast_obj) -> str:
    """Generates a base64 encoded PNG image of the AST."""
    try:
        graph = pydot.Dot("AST", graph_type="digraph", rankdir="TB")

        def agregar_nodo(nodo, padre=None):
            if isinstance(nodo, dict):
                label = nodo.get("_type", str(type(nodo)))
                current_node = pydot.Node(id(nodo), label=label, shape="box", style="rounded,filled", fillcolor="#E3F2FD")
                graph.add_node(current_node)

                if padre:
                    graph.add_edge(pydot.Edge(id(padre), id(nodo)))

                for k, v in nodo.items():
                    if isinstance(v, (dict, list)):
                        agregar_nodo(v, nodo)

            elif isinstance(nodo, list):
                for elemento in nodo:
                    agregar_nodo(elemento, padre)

            else:
                leaf_node = pydot.Node(id(nodo), label=str(nodo), shape="ellipse", fillcolor="#FFF9C4", style="filled")
                graph.add_node(leaf_node)
                if padre:
                    graph.add_edge(pydot.Edge(id(padre), id(nodo)))

        agregar_nodo(ast_obj.to_dict() if hasattr(ast_obj, "to_dict") else ast_obj)
        
        # Create PNG in memory
        png_data = graph.create_png()
        base64_encoded = base64.b64encode(png_data).decode("utf-8")
        return f"data:image/png;base64,{base64_encoded}"
    except Exception as e:
        print(f"Error generating AST image: {e}")
        return ""

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pseudocode(request: AnalysisRequest):
    if "parser" not in services or "analizador" not in services:
        raise HTTPException(status_code=500, detail="Services not initialized")

    try:
        pseudocode = request.pseudocode
        print(f"Analyzing pseudocode:\n{pseudocode}")

        # Step 1: Parse
        parser = services["parser"]
        ast_obj = parser.parsear(pseudocode)
        
        # Step 2: Analyze Complexity
        analizador = services["analizador"]
        algoritmo = Algoritmo(id=1, codigo_fuente=pseudocode, tipo_algoritmo=TipoAlgoritmo.ITERATIVO)
        algoritmo.addAST(ast_obj)
        
        resultado_complejidad = analizador.analizar(algoritmo)
        
        # Step 3: Validate with LLM
        llm_service = services["llm_service"]
        reporte = Reporte(id=1, algoritmo_analizado=algoritmo, resultado_complejidad=resultado_complejidad)
        validacion_ia = llm_service.validar_analisis(resultado_complejidad, pseudocode)
        
        # Step 4: Generate AST Image
        ast_image_base64 = generate_ast_image_base64(ast_obj)

        return AnalysisResponse(
            complexity_o=resultado_complejidad.notacion_o,
            complexity_omega=resultado_complejidad.notacion_omega,
            complexity_theta=resultado_complejidad.notacion_theta,
            justification=resultado_complejidad.justificacion_matematica,
            justification_data=resultado_complejidad.justification_data,
            validation=validacion_ia,
            ast_image=ast_image_base64
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)

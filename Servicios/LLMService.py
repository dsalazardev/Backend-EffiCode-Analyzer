from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional
from pathlib import Path

# Dependencias (asegúrate de tenerlas instaladas)
# Dependencias actualizadas para el nuevo SDK
# pip install python-dotenv "google-genai>1.0.0"
# pip install python-dotenv
# Usar en el main
# from dotenv import load_dotenv
# load_dotenv()

# Se importa la librería de Google para interactuar con Gemini

try:
    from google import genai
    from google.genai import errors as exceptions
    from google.genai.types import File
except ImportError:
    print("Dependencia no encontrada. Instale 'google-genai'.")
    genai = None
    File = None

if TYPE_CHECKING:
    from Modelos.Analizador import Analizador
    from Modelos.Complejidad import Complejidad
    from Modelos.Algoritmo import Algoritmo


class LLMService:

    """
    Servicio principal de IA refactorizado para el nuevo SDK google-genai.
    Opera con el contexto permanente del libro "Introduction to Algorithms" de Cormen,
    usando una estrategia de caché para optimizar el tiempo de inicio.
    """

    _ROOT = Path(__file__).resolve().parent.parent
    _RUTA_LIBRO = _ROOT / "Documentos" / "Introduction_to_Algorithms_by_Thomas_H_Coremen.pdf"
    _CACHE_FILE = _ROOT / "cache_file.json"

    def __init__(self, modelo: str = "gemini-2.5-pro"):

        """
        Inicializa el servicio LLM, configura el cliente de la API y gestiona la carga
        del libro de contexto usando un sistema de caché.
        """

        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("API Key no encontrada. Configure la variable de entorno 'GOOGLE_API_KEY'.")
        if not genai:
            raise RuntimeError("La librería 'google-genai' no está instalada.")

        self.client = genai.Client(api_key=self._api_key)
        self._modelo = modelo
        self._analizador: Optional[Analizador] = None

        self._libro_contexto_file = self._gestionar_cache_libro()


    def _gestionar_cache_libro(self) -> File:

        """
        Verifica si existe una referencia válida del libro en caché.
        Si no, sube el libro y guarda la nueva referencia en caché.
        """
        cache_data = self._cargar_cache()

        if cache_data:
            file_id = cache_data.get("file_id")
            display_name = cache_data.get("display_name", "Desconocido")
            upload_time_str = cache_data.get("upload_time")

            if file_id and upload_time_str:
                try:
                    upload_time = datetime.fromisoformat(upload_time_str)
                    if datetime.now() - upload_time < timedelta(hours=48):
                        print(
                            f"--- [LLMService] Intentando reutilizar archivo en caché: {display_name} ({file_id}) ---")
                        file_obj = self.client.files.get(name=file_id)

                        if getattr(file_obj, "state", None) == "ACTIVE":
                            print(f"Éxito. Usando el libro '{file_obj.display_name or display_name}' desde la caché.")
                            return file_obj
                        else:
                            print(f"El archivo '{display_name}' existe pero no está activo. Se volverá a subir.")
                    else:
                        print(f"El archivo en caché expiró (más de 48h). Se subirá nuevamente.")
                except exceptions.NotFound:
                    print(f"El archivo '{display_name}' ya no existe en Google. Se subirá de nuevo.")
                except Exception as e:
                    print(f"Error al validar caché: {e}. Se procederá a subir nuevamente.")

        return self._subir_y_guardar_libro_en_cache()


    def _cargar_cache(self) -> Optional[dict]:

        """Carga los datos del archivo de caché si existe."""

        if self._CACHE_FILE.exists():
            with self._CACHE_FILE.open('r') as f:
                return json.load(f)
        return None


    def _subir_y_guardar_libro_en_cache(self) -> File:

        """Sube el libro a la API de Google usando el nuevo SDK y guarda su referencia local en caché."""

        print("--- [LLMService] Subiendo el libro de contexto a la API de Archivos de Google... ---")

        if not self._RUTA_LIBRO.exists():
            raise FileNotFoundError(f"No se encontró el libro en la ruta interna: {self._RUTA_LIBRO}")

        try:
            print(f"Ruta del libro: {self._RUTA_LIBRO}")

            file_obj = self.client.files.upload(
                file=str(self._RUTA_LIBRO),
                config={"display_name": os.path.basename(self._RUTA_LIBRO)}
            )

            display_name = getattr(file_obj, "display_name", os.path.basename(self._RUTA_LIBRO))
            file_id = getattr(file_obj, "name", "N/A")
            size = getattr(file_obj, "size_bytes", 0)
            state = getattr(file_obj, "state", "DESCONOCIDO")
            uri = getattr(file_obj, "uri", "No disponible")

            print("✅ Archivo subido correctamente.")
            print(f"   - ID interno: {file_id}")
            print(f"   - Nombre mostrado: {display_name}")
            print(f"   - Tamaño: {size} bytes")
            print(f"   - Estado: {state}")
            print(f"   - URL (si aplica): {uri}")
            print(f"Libro '{display_name}' subido con éxito.")

            cache_data = {
                "file_id": file_id,
                "display_name": display_name,
                "upload_time": datetime.now().isoformat(),
                "path": str(self._RUTA_LIBRO)
            }
            with self._CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=4)

            print(f"--- [LLMService] Referencia guardada en '{self._CACHE_FILE}'. ---")
            return file_obj

        except Exception as e:
            raise RuntimeError(f"Error fatal al subir o guardar el libro: {str(e)}")


    def _ejecutar_prompt_con_contexto(self, prompt: str) -> str:

        """Función auxiliar que siempre incluye el libro como contexto."""

        try:
            response = self.client.models.generate_content(
                model=self._modelo,
                contents=[self._libro_contexto_file, prompt]
            )
            return response.text.strip()
        except Exception as e:
            # Devolver un marcador estándar para que el parser lo detecte
            err_msg = f"Error al contactar la API de Gemini: {e}"
            print(err_msg)
            return f"# Error: {err_msg}"


    def traducir_pseudocodigo_a_python(self, pseudocodigo: str) -> str:

        """Usa el LLM para convertir pseudocódigo a Python, basándose en el libro."""

        prompt = f"""
                    Actúa como un programador experto en algoritmos científico de la computación, especializado en traducir pseudocódigo del libro 'Introduction to Algorithms' de Cormen a Python idiomático.

                    **Contexto Académico:** Tu única fuente de verdad es el libro 'Introduction to Algorithms' proporcionado. 

                    **Tarea:** Basado en las convenciones del libro, traduce el siguiente pseudocódigo a una función de Python.

                    **Reglas Estrictas:**
                    1.  **Salida Exclusiva de Código:** Responde **únicamente con el código Python**.
                    2.  **Fidelidad al Libro:** Traduce `←` se convierte en `=`, `A.length` a `len(A)`, `≤, ≥, ≠` en `<=, >=, !=` y adapta los bucles de 1-indexado a 0-indexado.
                    3.  **Manejo de índices:** Adapta los bucles y accesos de 1-indexado (Cormen) a 0-indexado (Python).

                    **Pseudocódigo a Traducir:**
                    ---
                    {pseudocodigo}
                    ---
                    """
        codigo_generado = self._ejecutar_prompt_con_contexto(prompt)
        return codigo_generado.replace("```python", "").replace("```", "").strip()


    def traducir_natural_a_pseudocodigo(self, texto: str) -> str:

        """Usa el LLM para convertir lenguaje natural a pseudocódigo estilo Cormen."""

        prompt = f"""
                    Actúa como un experto en el libro 'Introduction to Algorithms' de Cormen.

                    **Contexto Académico:** Tu única fuente de verdad es el libro 'Introduction to Algorithms' proporcionado. 

                    **Tarea:** Convierte la siguiente descripción a pseudocódigo, siguiendo estrictamente el estilo del libro.

                    **Reglas de Estilo del Libro:**
                    - Usa `←` para asignaciones.
                    - Usa `A.length` para la longitud de arreglos.
                    - Usa bucles `for`, `while` con la sintaxis del libro.
                    - Usa comentarios con `//`.

                    **Descripción a Convertir que viene en lenguaje Natural:**
                    "{texto}"
                    """
        return self._ejecutar_prompt_con_contexto(prompt)


    def validar_analisis(self, complejidad: Complejidad, pseudocodigo: str) -> str:

        """Pide al LLM que valide un análisis de complejidad, usando el libro como referencia."""

        prompt = f"""
                Actúa como un experto en análisis de algoritmos, al nivel de un profesor de ciencias de la computación.

                **Contexto Académico:** Tu única fuente de verdad es el libro 'Introduction to Algorithms' proporcionado.

                **Tarea:** Como un profesor de algoritmos, revisa y valida el siguiente análisis de complejidad basándote en la teoría del libro.

                **Pseudocódigo:**
                {pseudocodigo}


                **Análisis Propuesto:**
                - O(n): {complejidad.notacion_o}
                - Ω(n): {complejidad.notacion_omega}
                - Justificación: {complejidad.justificacion_matematica}

                **Tu Respuesta:** Proporciona una segunda opinión experta y concisa. Confirma si es correcto o explica claramente cualquier error o matiz, citando conceptos del libro si es relevante.
                """
        return self._ejecutar_prompt_con_contexto(prompt)


    def clasificar_patron(self, algoritmo: Algoritmo) -> str:

        """Usa el LLM para identificar el patrón de diseño del algoritmo, según las definiciones del libro."""

        prompt = f"""
                Actúa como un científico de la computación experto en paradigmas de diseño de algoritmos.

                **Contexto Académico:** Tu única fuente de verdad es el libro 'Introduction to Algorithms' proporcionado.

                **Tarea:** Identifica el principal paradigma de diseño algorítmico del pseudocódigo, usando la terminología del libro.

                **Pseudocódigo:**
                {algoritmo.codigo_fuente}


                **Instrucciones:** Responde únicamente con el nombre del patrón (ej: 'Divide and conquer', 'Dynamic programming', 'Greedy algorithm').
                 Sé específico. Ejemplos: 'Divide y Vencerás', 'Programación Dinámica', 'Algoritmo Voraz'.
                 Si no identificas un patrón claro, responde 'No se identifica un patrón estándar'.
                """
        return self._ejecutar_prompt_con_contexto(prompt)
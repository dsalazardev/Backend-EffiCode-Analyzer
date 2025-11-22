import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk  # Importar ttk para widgets modernos
import threading
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# --- Importaciones de tu proyecto EffiCode Analyzer ---
from dotenv import load_dotenv
from Modelos.Parser import Parser
from Modelos.Analizador import Analizador
from Servicios.LLMService import LLMService
from Servicios.Grammar import Grammar
from Modelos.Algoritmo import Algoritmo
from Enumerations.tipoAlgoritmo import TipoAlgoritmo
from Modelos.Reporte import Reporte


class EffiCodeApp:
    def __init__(self, root):
        self.root = root
        root.title("EffiCode Analyzer")
        root.geometry("1200x800")
        
        # Configurar estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Un tema más limpio que el default
        
        # Colores y fuentes
        bg_color = "#f0f0f0"
        primary_color = "#2196F3"
        text_color = "#333333"
        
        self.root.configure(bg=bg_color)
        
        # Configurar estilos personalizados
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=text_color, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1565C0")
        self.style.configure("Result.TLabel", font=("Segoe UI", 12, "bold"))
        
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("TButton",
            foreground=[('pressed', 'white'), ('active', 'white')],
            background=[('pressed', '!disabled', '#0D47A1'), ('active', '#1976D2')]
        )
        
        self.style.configure("Accent.TButton", background=primary_color, foreground="white")

        # --- Inicializar los servicios del backend ---
        try:
            env_path = os.path.join(project_root, '.env')
            if os.path.exists(env_path):
                load_dotenv(dotenv_path=env_path)
            else:
                print("Advertencia: No se encontró el archivo .env en la raíz del proyecto.")

            self.grammar = Grammar()
            self.llm_service = LLMService()
            self.parser = Parser(id=1, gramatica=self.grammar, llm_service=self.llm_service)
            self.analizador = Analizador(id=1, parser=self.parser, llm_service=self.llm_service)
        except Exception as e:
            messagebox.showerror("Error de Inicialización",
                                 f"No se pudieron cargar los servicios del backend:\n{e}")
            root.quit()
            return

        # --- Diseño de la GUI ---
        main_container = ttk.Frame(root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="EffiCode Analyzer", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header_frame, text=" | Análisis de Complejidad Algorítmica", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=5)

        # PanedWindow para dividir entrada y salida
        self.paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # --- PANEL IZQUIERDO (Entrada) ---
        left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Pseudocódigo (Estilo Cormen)", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Consolas", 11), height=20, bd=1, relief="solid")
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.insert(tk.END,
                               "// Escribe tu algoritmo aquí\nINSERTION-SORT(A, n)\n    for j ← 2 to n do\n        key ← A[j]\n        i ← j - 1\n        while i > 0 and A[i] > key do\n            A[i+1] ← A[i]\n            i ← i - 1\n        A[i+1] ← key\n")

        # --- PANEL DERECHO (Resultados) ---
        right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_frame, weight=1)

        # Sección de Botón y Métricas
        metrics_frame = ttk.LabelFrame(right_frame, text="Resultados del Análisis", padding="10")
        metrics_frame.pack(fill=tk.X, padx=(5, 0), pady=(0, 10))

        self.analyze_button = ttk.Button(metrics_frame, text="▶ Analizar Complejidad", style="Accent.TButton",
                                     command=self.iniciar_analisis_thread)
        self.analyze_button.pack(fill=tk.X, pady=(0, 15))

        # Grid de resultados
        results_grid = ttk.Frame(metrics_frame)
        results_grid.pack(fill=tk.X)
        
        # Configurar columnas del grid
        results_grid.columnconfigure(0, weight=1)
        results_grid.columnconfigure(1, weight=1)
        results_grid.columnconfigure(2, weight=1)

        # Peor Caso
        f1 = ttk.Frame(results_grid, padding=5, relief="flat")
        f1.grid(row=0, column=0, sticky="nsew")
        ttk.Label(f1, text="Peor Caso (O)", font=("Segoe UI", 9)).pack()
        self.o_label = ttk.Label(f1, text="-", style="Result.TLabel", foreground="#D32F2F")
        self.o_label.pack()

        # Mejor Caso
        f2 = ttk.Frame(results_grid, padding=5, relief="flat")
        f2.grid(row=0, column=1, sticky="nsew")
        ttk.Label(f2, text="Mejor Caso (Ω)", font=("Segoe UI", 9)).pack()
        self.omega_label = ttk.Label(f2, text="-", style="Result.TLabel", foreground="#1976D2")
        self.omega_label.pack()

        # Caso Promedio
        f3 = ttk.Frame(results_grid, padding=5, relief="flat")
        f3.grid(row=0, column=2, sticky="nsew")
        ttk.Label(f3, text="Caso Promedio (Θ)", font=("Segoe UI", 9)).pack()
        self.theta_label = ttk.Label(f3, text="-", style="Result.TLabel", foreground="#388E3C")
        self.theta_label.pack()

        # Pestañas de Detalles
        self.tab_control = ttk.Notebook(right_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=(5, 0))

        self.trace_tab = ttk.Frame(self.tab_control)
        self.validation_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.trace_tab, text=' 🔍 Traza Matemática ')
        self.tab_control.add(self.validation_tab, text=' 🤖 Validación IA ')

        self.trace_output = scrolledtext.ScrolledText(self.trace_tab, wrap=tk.WORD, state='disabled',
                                                      font=("Consolas", 10), bd=0)
        self.trace_output.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.validation_output = scrolledtext.ScrolledText(self.validation_tab, wrap=tk.WORD, state='disabled',
                                                           font=("Consolas", 10), bd=0)
        self.validation_output.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    def update_output(self, text_widget, message):
        self.root.after(0, self._update_text_widget, text_widget, message)

    def _update_text_widget(self, text_widget, message):
        text_widget.config(state='normal')
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, str(message))
        text_widget.config(state='disabled')

    def update_label(self, label, text):
        self.root.after(0, lambda: label.config(text=text))

    def update_button(self, state, text):
        self.root.after(0, lambda: self.analyze_button.config(state=state, text=text))

    def iniciar_analisis_thread(self):
        self.update_button(state="disabled", text="⏳ Analizando...")
        
        self.update_label(self.o_label, "Calculando...")
        self.update_label(self.omega_label, "Calculando...")
        self.update_label(self.theta_label, "Calculando...")

        self.update_output(self.trace_output, "Iniciando análisis...")
        self.update_output(self.validation_output, "Esperando análisis matemático...")

        pseudocode = self.input_text.get("1.0", tk.END)
        analysis_thread = threading.Thread(target=self.run_analysis_backend, args=(pseudocode,), daemon=True)
        analysis_thread.start()

    def run_analysis_backend(self, pseudocode):
        try:
            # --- [PASO 1] PARSER ---
            self.update_output(self.trace_output, "Paso 1: Parseando y traduciendo pseudocódigo a AST...")
            ast_obj = self.parser.parsear(pseudocode)

            # --- [PASO 2] ANÁLISIS MATEMÁTICO ---
            self.update_output(self.trace_output, "Paso 2: Ejecutando análisis de eficiencia matemática...")
            algoritmo = Algoritmo(id=1, codigo_fuente=pseudocode, tipo_algoritmo=TipoAlgoritmo.ITERATIVO)
            algoritmo.addAST(ast_obj)

            resultado_complejidad = self.analizador.analizar(algoritmo)
            self.update_output(self.trace_output, resultado_complejidad.justificacion_matematica)

            # --- [PASO 3] VALIDACIÓN IA ---
            self.update_output(self.validation_output, "Paso 3: Solicitando validación del análisis a la IA...")
            reporte = Reporte(id=1, algoritmo_analizado=algoritmo, resultado_complejidad=resultado_complejidad)
            validacion_ia = self.llm_service.validar_analisis(resultado_complejidad, pseudocode)
            reporte.validacion_llm = validacion_ia

            # --- [PASO 4] MOSTRAR RESULTADOS ---
            self.update_label(self.o_label, reporte.resultado_complejidad.notacion_o)
            self.update_label(self.omega_label, reporte.resultado_complejidad.notacion_omega)
            self.update_label(self.theta_label, reporte.resultado_complejidad.notacion_theta)
            self.update_output(self.validation_output, reporte.validacion_llm)

        except Exception as e:
            import traceback
            error_msg = f"❌ ERROR:\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            self.update_output(self.trace_output, error_msg)
            self.update_output(self.validation_output, "Análisis fallido.")
            self.update_label(self.o_label, "Error")
            self.update_label(self.omega_label, "Error")
            self.update_label(self.theta_label, "Error")
            self.root.after(0, lambda: messagebox.showerror("Error de Análisis", f"Ocurrió un error durante el análisis:\n{e}"))

        finally:
            self.update_button(state="normal", text="▶ Analizar Complejidad")


if __name__ == "__main__":
    root = tk.Tk()
    app = EffiCodeApp(root)
    root.mainloop()

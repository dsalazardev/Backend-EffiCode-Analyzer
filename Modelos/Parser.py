from __future__ import annotations
from typing import TYPE_CHECKING
import re

from Servicios.Grammar import Grammar
from Servicios.LLMService import LLMService
from Servicios.Ast import AST

# Importaciones de tipos para evitar dependencias circulares
if TYPE_CHECKING:
    from .Analizador import Analizador

class Parser:

    def __init__(self, id: int, gramatica: Grammar, llm_service: LLMService):
        self._id = id
        self._gramatica = gramatica
        self._llm_service = llm_service
        self._analizador: Analizador | None = None

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def gramatica(self) -> Grammar:
        return self._gramatica

    @gramatica.setter
    def gramatica(self, value: Grammar):
        self._gramatica = value

    @property
    def analizador(self) -> Analizador | None:
        return self._analizador

    @analizador.setter
    def analizador(self, value: Analizador | None):
        self._analizador = value

    def addAnalizador(self, analizador: Analizador):
        self._analizador = analizador

    def removeAnalizador(self, analizador: Analizador):
        if self._analizador == analizador:
            self._analizador = None

    def validar_sintaxis(self, pseudocodigo: str) -> bool:
        """
        Verifica si el código cumple con la gramática definida.
        Delega la validación al servicio de Gramática.
        """
        return self._gramatica.validar_sentencia(pseudocodigo)

    def parsear(self, pseudocodigo: str) -> AST:
        """
        Convierte el pseudocódigo en un Árbol de Sintaxis Abstracta (AST) de Python.
        
        ESTRATEGIA: Usa un traductor LOCAL robusto primero, sin depender del LLM.
        El LLM solo se usa como validación opcional.

        Args:
            pseudocodigo (str): El código fuente en pseudocódigo estilo Cormen.

        Returns:
            AST: Una instancia del servicio AST que contiene el árbol del código Python.

        Raises:
            SyntaxError: Si la sintaxis del pseudocódigo es inválida o si el
                         código Python generado es inválido.
        """
        # 1. Validar la sintaxis usando el servicio de Gramática (opcional, puede ser flexible)
        # No bloqueamos si falla, intentamos traducir de todos modos
        syntax_valid = self.validar_sintaxis(pseudocodigo)
        if not syntax_valid:
            print("⚠️  Advertencia: El pseudocódigo puede no cumplir estrictamente la gramática Cormen.")

        # 2. Traducir a Python usando el traductor LOCAL (no depende de LLM)
        codigo_python = self._translate_pseudocode_to_python(pseudocodigo)
        
        if not codigo_python or not codigo_python.strip():
            raise SyntaxError("Error: No se pudo traducir el pseudocódigo a Python.")

        print(f"--- Código Python generado ---\n{codigo_python}\n--- Fin código Python ---")

        # 3. Parsear el código Python y devolver el objeto AST
        try:
            ast_obj = AST(codigo_python)
            return ast_obj
        except SyntaxError as e:
            raise SyntaxError(
                f"El código Python generado no es válido. Error: {e}\nCódigo generado:\n{codigo_python}")

    def _translate_pseudocode_to_python(self, pseudocodigo: str) -> str:
        """
        Traductor LOCAL robusto de pseudocódigo Cormen a Python.
        
        Soporta:
        - for i ← start to end do  →  for i in range(start-1, end):
        - for i ← start downto end do  →  for i in range(start, end-1, -1):
        - while cond do  →  while cond:
        - if cond then  →  if cond:
        - else  →  else:
        - x ← expr  →  x = expr
        - return expr  →  return expr
        - // comentarios  →  # comentarios
        - FUNC(args) llamadas  →  func(args)
        - Operadores: ≤ → <=, ≥ → >=, ≠ → !=, and/or/not
        - A[i] acceso a arrays (se mantiene)
        """
        lines = pseudocodigo.split('\n')
        python_lines = []
        
        for line in lines:
            # Preservar indentación original
            original_indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            
            # Línea vacía
            if not stripped:
                python_lines.append('')
                continue
            
            # Comentarios en línea completa
            if stripped.startswith('//'):
                python_lines.append(' ' * original_indent + '#' + stripped[2:])
                continue
            
            # Separar comentarios inline (// ...) del código
            code_part = stripped
            comment_part = ''
            if '//' in stripped:
                idx = stripped.index('//')
                code_part = stripped[:idx].strip()
                comment_part = '  #' + stripped[idx+2:]
            
            # Si solo quedó comentario, continuar
            if not code_part:
                python_lines.append(' ' * original_indent + comment_part.strip())
                continue
            
            # Detectar declaración de función vs llamada a función
            # REGLA SIMPLE: Solo es DECLARACIÓN si está en indentación 0 (inicio de línea)
            # Cualquier NOMBRE(args) con indentación es una LLAMADA
            func_pattern = r'^([A-Z][A-Z0-9_-]*)\s*\(([^)]*)\)\s*$'
            match = re.match(func_pattern, code_part)
            
            if match:
                func_name = match.group(1).replace('-', '_').lower()
                params = match.group(2)
                
                if original_indent == 0:
                    # Declaración de función (sin indentación)
                    python_lines.append(f'def {func_name}({params}):')
                else:
                    # Llamada a función (tiene indentación)
                    python_lines.append(' ' * original_indent + f'{func_name}({params})' + comment_part)
                continue
            
            # Traducir la línea de código
            translated = self._translate_line(code_part)
            
            # Agregar comentario inline si existe
            python_lines.append(' ' * original_indent + translated + comment_part)
        
        # Limpiar y normalizar indentación
        return self._normalize_indentation(python_lines)

    def _translate_line(self, line: str) -> str:
        """Traduce una línea individual de pseudocódigo a Python."""
        
        # Reemplazar operadores especiales primero
        line = line.replace('≤', '<=')
        line = line.replace('≥', '>=')
        line = line.replace('≠', '!=')
        line = line.replace('←', '=')
        
        # FOR ... TO ... DO (ascendente)
        match = re.match(
            r'^for\s+(\w+)\s*=\s*(.+?)\s+to\s+(.+?)\s+do\s*$',
            line, re.IGNORECASE
        )
        if match:
            var, start, end = match.groups()
            start = start.strip()
            end = end.strip()
            # Ajustar para Python (range es exclusivo en el límite superior)
            return f'for {var} in range({start}, {end} + 1):'
        
        # FOR ... DOWNTO ... DO (descendente)
        match = re.match(
            r'^for\s+(\w+)\s*=\s*(.+?)\s+downto\s+(.+?)\s+do\s*$',
            line, re.IGNORECASE
        )
        if match:
            var, start, end = match.groups()
            start = start.strip()
            end = end.strip()
            # range(start, end-1, -1) para incluir end
            return f'for {var} in range({start}, {end} - 1, -1):'
        
        # WHILE ... DO
        match = re.match(r'^while\s+(.+?)\s+do\s*$', line, re.IGNORECASE)
        if match:
            condition = match.group(1).strip()
            return f'while {condition}:'
        
        # IF ... THEN
        match = re.match(r'^if\s+(.+?)\s+then\s*$', line, re.IGNORECASE)
        if match:
            condition = match.group(1).strip()
            return f'if {condition}:'
        
        # ELSE IF ... THEN
        match = re.match(r'^else\s+if\s+(.+?)\s+then\s*$', line, re.IGNORECASE)
        if match:
            condition = match.group(1).strip()
            return f'elif {condition}:'
        
        # ELSE (solo)
        if line.lower().strip() == 'else':
            return 'else:'
        
        # RETURN
        match = re.match(r'^return\s+(.+)$', line, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            return f'return {expr}'
        
        # RETURN sin valor
        if line.lower().strip() == 'return':
            return 'return'
        
        # Llamada a función sola en la línea (ej: QUICKSORT(A, p, q - 1))
        # Esto es una LLAMADA, no una declaración
        match = re.match(r'^([A-Z][A-Z0-9_-]*)\s*\((.+)\)\s*$', line)
        if match:
            func_name = match.group(1).replace('-', '_').lower()
            args = match.group(2)
            return f'{func_name}({args})'
        
        # Convertir llamadas a funciones en MAYÚSCULAS dentro de expresiones
        # (ej: q = PARTITION(A, p, r) → q = partition(A, p, r))
        def replace_func_call(m):
            func_name = m.group(1).replace('-', '_').lower()
            args = m.group(2)
            return f'{func_name}({args})'
        
        line = re.sub(r'\b([A-Z][A-Z0-9_-]*)\s*\(([^)]*)\)', replace_func_call, line)
        
        return line

    def _normalize_indentation(self, lines: list) -> str:
        """
        Normaliza la indentación del código Python generado.
        Convierte la indentación basada en espacios del pseudocódigo
        a indentación Python estándar de 4 espacios.
        """
        result = []
        indent_stack = [0]  # Pila de niveles de indentación
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result.append('')
                continue
            
            # Calcular indentación actual
            current_indent = len(line) - len(line.lstrip())
            
            # Determinar el nivel de indentación Python
            # Comparar con el nivel anterior
            while indent_stack and current_indent < indent_stack[-1]:
                indent_stack.pop()
            
            if current_indent > indent_stack[-1]:
                indent_stack.append(current_indent)
            
            # Calcular nivel Python (cada nivel = 4 espacios)
            python_level = len(indent_stack) - 1
            python_indent = '    ' * python_level
            
            result.append(python_indent + stripped)
        
        return '\n'.join(result)

    def _heuristic_translate(self, pseudocodigo: str) -> str:
        """Fallback - usa el nuevo traductor."""
        return self._translate_pseudocode_to_python(pseudocodigo)

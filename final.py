import customtkinter as ctk
from tkinter import filedialog, messagebox
from io import BytesIO
import re
import pydot
import base64
from PIL import Image, ImageTk
############################################
# CONFIGURACIÓN DE CUSTOMTKINTER
############################################
ctk.set_appearance_mode("System")  # "Dark", "Light" o "System"
ctk.set_default_color_theme("blue")  # Puedes cambiar el tema (blue, dark-blue, green)

############################################
# ANALIZADOR LÉXICO
############################################

TOKEN_PATTERNS = [
    (r'\b(true|false|null)\b', 'LITERAL_BOOLEANO_NULO'),
    (r'\b(abstract|assert|boolean|break|byte|case|catch|char|class|const|'
     r'continue|default|do|double|else|enum|extends|final|finally|float|for|'
     r'goto|if|implements|import|instanceof|int|interface|long|native|new|'
     r'package|private|protected|public|return|short|static|strictfp|super|'
     r'switch|synchronized|this|throw|throws|transient|try|var|void|volatile|'
     r'while|record|sealed|non-sealed|yield|module|open|opens|requires|'
     r'exports|provides|to|with|uses|transitive|permits|String)\b', 'PALABRA_RESERVADA'),
    (r'>>>=|>>=|<<=|\+\+|--|->|>=|<=|==|!=|&&|\|\||<<|>>>|>>|::|\.\.\.|'
     r'[+\-*/%&|^<>]=?|~|\?|:|@|={1,3}|!|\+|\-|\*|/|%|&|\||\^|<|>|\(|\)|\[|\]|\{|\}|;|,|\.', 'OPERADOR/SEPARADOR'),
    (r'0[xX][0-9a-fA-F_]+[lL]?', 'LITERAL_HEXADECIMAL'),
    (r'0[bB][01_]+[lL]?', 'LITERAL_BINARIO'),
    (r'0[0-7_]+[lL]?', 'LITERAL_OCTAL'),
    (r'((\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d[\d_]*)?[fFdD]?)|(\d[\d_]*[eE][+-]?\d[\d_]*[fFdD]?)|(\d[\d_]*[fFdD])', 'LITERAL_FLOTANTE'),
    (r'\d[\d_]*[lL]?', 'LITERAL_ENTERO'),
    (r'"(?:\\.|[^"\\])*"', 'LITERAL_CADENA'),
    (r"'(\\\\|\\'|[^']|\\u[0-9a-fA-F]{4})'", 'LITERAL_CARACTER'),
    (r'System\s*\.\s*out\s*\.\s*println', 'FUNCION_IMPRESION'),
    (r'\b(public|class|static|void|main|String\s*\[\s*\]|args)\b', 'IGNORAR'),
    (r'[a-zA-Z_$][a-zA-Z0-9_$]*', 'IDENTIFICADOR'),
]

IGNORE_PATTERNS = [
    r'\s+',
    r'//.*',
    r'/\*(.|\n)*?\*/',
    r'public\s+class\s+\w+\s*\{',  # Ignora 'public class Main {'
    r'public\s+static\s+void\s+main\s*\(String\[\]\s*\w*\s*\)\s*\{',  # Ignora 'public static void main(String[] args) {'
    r'\}',  # Ignora llaves de cierre
]

TOKEN_REGEX = [(re.compile(pattern), token_type) for pattern, token_type in TOKEN_PATTERNS]
IGNORE_REGEX = [re.compile(pattern) for pattern in IGNORE_PATTERNS]

def analizador_lexico(codigo_fuente):
    tokens = []
    errores = []
    lineas = codigo_fuente.split('\n')
    posicion = 0
    num_linea = 1
    while posicion < len(codigo_fuente):
        # Ignorar espacios y comentarios
        ignorado = False
        for regex in IGNORE_REGEX:
            match = regex.match(codigo_fuente, posicion)
            if match:
                if '\n' in match.group():
                    num_linea += match.group().count('\n')
                posicion = match.end()
                ignorado = True
                break
        if ignorado:
            continue

        # Buscar tokens válidos
        coincidencia = None
        for regex, tipo in TOKEN_REGEX:
            match = regex.match(codigo_fuente, posicion)
            if match:
                coincidencia = (match.group(), tipo)
                tokens.append(coincidencia)
                posicion = match.end()
                break

        # Error léxico si no se encontró token válido
        if not coincidencia:
            columna = posicion - sum(len(lineas[i]) + 1 for i in range(num_linea - 1))
            caracter = codigo_fuente[posicion]
            tipo_error = "Carácter inválido"
            if caracter in {'"', "'"}:
                tipo_error = "Cadena o carácter no cerrado"
            elif caracter in {'(', '{', '['}:
                tipo_error = "Paréntesis no cerrado"
            elif caracter.isdigit():
                tipo_error = "Número incorrecto"
            elif caracter.isalpha():
                tipo_error = "Identificador mal formado"
            errores.append((caracter, num_linea, columna + 1, tipo_error))
            posicion += 1
    return tokens, errores, lineas

############################################
# ESTRUCTURAS PARA EL AST Y TABLA DE SÍMBOLOS
############################################

class ASTNode:
    """Clase base para los nodos del árbol sintáctico."""
    def __init__(self, nodetype, children=None, value=None):
        self.nodetype = nodetype
        self.children = children or []
        self.value = value
        self.id = id(self)  # ID único

    def __str__(self, level=0):
        ret = "  " * level + f"{self.nodetype}"
        if self.value is not None:
            ret += f": {self.value}"
        ret += "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

class SymbolTable:
    def __init__(self):
        self.symbols = {}  # Diccionario: {nombre_variable: tipo}
        self.conversiones_permitidas = {
            'int': ['float', 'double', 'String'],
            'float': ['double', 'String'],
            'char': ['String'],
            'String': []  # Añadir String como tipo
        }

    def declare(self, identifier, tipo):
        """Registra una nueva variable en la tabla de símbolos"""
        if identifier in self.symbols:
            return False  # Variable ya declarada
        self.symbols[identifier] = tipo
        return True

    def exists(self, identifier):
        """Verifica si una variable está declarada"""
        return identifier in self.symbols

    def get_type(self, identifier):
        """Obtiene el tipo de una variable"""
        return self.symbols.get(identifier, None)

    def is_conversion_allowed(self, from_type, to_type):
        """Verifica si una conversión implícita es permitida"""
        if from_type == to_type:
            return True
        return to_type in self.conversiones_permitidas.get(from_type, [])

############################################
# ANALIZADOR SINTÁCTICO Y GENERACIÓN DE CÓDIGO INTERMEDIO
############################################

class Parser:
    """
    Analizador sintáctico basado en recursive descent para un subconjunto de Java.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.errores = []
        self.symbol_table = SymbolTable()
        self.codigo_intermedio = []
        self.temp_counter = 0

    def token_actual(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consumir(self, expected_value=None, expected_tipo=None):
        token = self.token_actual()
        if token is None:
            return None  # Simplemente retorna None en lugar de registrar error
            
        lexema, tipo = token
        
        # Ignorar llaves y estructura de clase
        if lexema in ['{', '}']:
            self.pos += 1
            return None
            
        # Resto de la lógica original...
        if expected_value and lexema != expected_value:
            self.errores.append(f"Se esperaba '{expected_value}' pero se encontró '{lexema}'")
            return None
        if expected_tipo and tipo != expected_tipo:
            self.errores.append(f"Se esperaba un token de tipo {expected_tipo} pero se encontró {tipo}")
            return None
            
        self.pos += 1
        return token

    def nuevo_temp(self):
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def parse(self):
        nodo_programa = ASTNode("Program")
        while self.token_actual() is not None:
            stmt = self.parse_statement()
            if stmt:
                nodo_programa.children.append(stmt)
            else:
                self.pos += 1
        return nodo_programa

    def parse_statement(self):
        token = self.token_actual()
        if token is None:
            return None
        lexema, tipo = token

        # Manejar if statements
        if lexema == "if":
            self.consumir(expected_value="if")
            self.consumir(expected_value="(")
            cond_node, temp = self.parse_expression()
            self.consumir(expected_value=")")
            self.consumir(expected_value="{")
            self.codigo_intermedio.append(f"if {temp}:")
            nodo_if = ASTNode("IfStatement", children=[cond_node])
            
            # Parsear bloque interno
            while self.token_actual() and self.token_actual()[0] != "}":
                stmt = self.parse_statement()
                if stmt:
                    nodo_if.children.append(stmt)
            
            self.consumir(expected_value="}")
            return nodo_if

        # Manejar System.out.println como caso especial
        if tipo == "FUNCION_IMPRESION":
            self.consumir(expected_tipo="FUNCION_IMPRESION")
            self.consumir(expected_value="(")
            expr_node, temp = self.parse_expression()
            self.consumir(expected_value=")")
            self.consumir(expected_value=";")
            self.codigo_intermedio.append(f"print({temp})")
            return ASTNode("Print", children=[expr_node])
            
        # Manejar declaraciones de variables (incluyendo String)
        if (tipo == "PALABRA_RESERVADA" and lexema in {'int', 'double', 'float', 'String', 'boolean', 'char'}) or \
        (lexema == "String"):  # Añadir String como caso especial
            tipo_variable = lexema
            self.consumir()  # Consumir el tipo de dato
            
            # Manejar arrays (String[])
            if self.token_actual() and self.token_actual()[0] == '[':
                self.consumir(expected_value='[')
                self.consumir(expected_value=']')
                tipo_variable += '[]'
                
            id_token = self.consumir(expected_tipo="IDENTIFICADOR")
            if id_token is None:
                return None
                
            identificador = id_token[0]
            nodo_decl = ASTNode("Declaracion", value=identificador)
            
            if not self.symbol_table.declare(identificador, tipo_variable):
                self.errores.append(f"Variable '{identificador}' ya declarada")
                
            token = self.token_actual()
            if token and token[0] == "=":
                self.consumir(expected_value="=")
                expr_node, temp = self.parse_expression()
                nodo_decl.children.append(expr_node)
                self.codigo_intermedio.append(f"{identificador} = {temp}")
                
            self.consumir(expected_value=";")
            return nodo_decl

        elif tipo == "IDENTIFICADOR":
            id_token = self.consumir(expected_tipo="IDENTIFICADOR")
            identificador = id_token[0]
            nodo_asig = ASTNode("Asignacion", value=identificador)
            
            if not self.symbol_table.exists(identificador):
                self.errores.append(f"Variable '{identificador}' no declarada")
                
            token = self.token_actual()
            if token and token[0] == "=":
                self.consumir(expected_value="=")
                expr_node, temp = self.parse_expression()
                nodo_asig.children.append(expr_node)
                self.codigo_intermedio.append(f"{identificador} = {temp}")
                self.consumir(expected_value=";")
                return nodo_asig
            else:
                expr_node, temp = self.parse_expression()
                self.consumir(expected_value=";")
                return expr_node
        else:
            expr_node, temp = self.parse_expression()
            self.consumir(expected_value=";")
            return expr_node

    def parse_expression(self):
        nodo, temp = self.parse_term()
        while self.token_actual() and self.token_actual()[0] in ("+", "-", ">", "<", ">=", "<=", "==", "!="):
            op = self.token_actual()[0]
            self.consumir(expected_value=op)
            nodo_der, temp_der = self.parse_term()
            nuevo_temp = self.nuevo_temp()
            nodo = ASTNode("Operacion", children=[nodo, nodo_der], value=op)
            self.codigo_intermedio.append(f"{nuevo_temp} = {temp} {op} {temp_der}")
            temp = nuevo_temp
        return nodo, temp

    def parse_term(self):
        nodo, temp = self.parse_factor()
        while self.token_actual() and self.token_actual()[0] in ("*", "/"):
            op = self.token_actual()[0]
            self.consumir(expected_value=op)
            nodo_der, temp_der = self.parse_factor()
            nuevo_temp = self.nuevo_temp()
            nodo = ASTNode("Operacion", children=[nodo, nodo_der], value=op)
            self.codigo_intermedio.append(f"{nuevo_temp} = {temp} {op} {temp_der}")
            temp = nuevo_temp
        return nodo, temp

    def parse_factor(self):
        token = self.token_actual()
        if token is None:
            self.errores.append("Se esperaba un factor pero se llegó al final de la entrada")
            return ASTNode("Error"), "error"
        lexema, tipo = token

        if lexema == "(":
            self.consumir(expected_value="(")
            nodo, temp = self.parse_expression()
            self.consumir(expected_value=")")
            return nodo, temp
        elif tipo in ("LITERAL_ENTERO", "LITERAL_FLOTANTE", "LITERAL_CADENA", "LITERAL_CARACTER", "LITERAL_BOOLEANO_NULO"):
            self.consumir()
            temp = self.nuevo_temp()
            self.codigo_intermedio.append(f"{temp} = {lexema}")
            return ASTNode("Literal", value=lexema), temp
        elif tipo == "IDENTIFICADOR":
            self.consumir(expected_tipo="IDENTIFICADOR")
            temp = self.nuevo_temp()
            if not self.symbol_table.exists(lexema):
                self.errores.append(f"Variable '{lexema}' no declarada")
            self.codigo_intermedio.append(f"{temp} = {lexema}")
            return ASTNode("Identificador", value=lexema), temp
        elif tipo == "FUNCION_IMPRESION":
            # Esto no debería ocurrir aquí, ya que se maneja en parse_statement
            self.errores.append(f"Token inesperado en factor: {lexema}")
            self.consumir()
            return ASTNode("Error", value=lexema), "error"
        else:
            self.errores.append(f"Token inesperado: {lexema}")
            self.consumir()
            return ASTNode("Error", value=lexema), "error"

############################################
# FUNCIÓN PARA CONSTRUIR EL GRAFO DEL AST CON PYDOT
############################################

def add_nodes_edges(node, graph):
    """
    Función recursiva que agrega nodos y aristas al grafo de pydot.
    """
    node_label = f"{node.nodetype}"
    if node.value is not None:
        node_label += f": {node.value}"
    graph_node = pydot.Node(str(node.id), label=node_label)
    graph.add_node(graph_node)
    for child in node.children:
        add_nodes_edges(child, graph)
        graph.add_edge(pydot.Edge(str(node.id), str(child.id)))

def generar_ast_graph(root):
    """
    Crea y retorna un objeto pydot.Dot que representa el grafo del AST.
    """
    graph = pydot.Dot(graph_type='graph')
    add_nodes_edges(root, graph)
    return graph

############################################
# FUNCIÓN DE CONVERSIÓN JAVA A PYTHON
############################################

def convert_java_to_python(java_code):
    # Eliminar estructura de clase y main preservando el contenido
    java_code = re.sub(r'public\s+class\s+\w+\s*\{', '', java_code)
    java_code = re.sub(r'public\s+static\s+void\s+main\s*\(String\[\]\s*\w*\)\s*\{', '', java_code)
    java_code = re.sub(r'\}\s*$', '', java_code)  # Eliminar llave de cierre
    
    # Convertir System.out.println a print
    java_code = re.sub(r'System\.out\.println\((.*?)\);', r'print(\1)', java_code)
    
    # Convertir booleanos a Python
    java_code = re.sub(r'\btrue\b', 'True', java_code)
    java_code = re.sub(r'\bfalse\b', 'False', java_code)
    
    # Manejar condicionales if
    java_code = re.sub(r'\bif\s*\((.*?)\)\s*\{', r'if \1:', java_code)
    
    # Manejar else if y else
    java_code = re.sub(r'\belse\s+if\s*\((.*?)\)\s*\{', r'elif \1:', java_code)
    java_code = re.sub(r'\belse\s*\{', r'else:', java_code)
    
    # Eliminar comentarios multi-línea
    java_code = re.sub(r'/\*.*?\*/', '', java_code, flags=re.DOTALL)
    
    lines = java_code.split('\n')
    new_lines = []
    indent_level = 0
    in_block = False
    
    for line in lines:
        # Eliminar llaves y punto y coma
        line = line.replace('{', '').replace('}', '').replace(';', '').strip()
        
        # Saltar líneas vacías
        if not line:
            continue
        
        # Manejar disminución de indentación
        if line.startswith('}') or line.endswith('}'):
            indent_level = max(0, indent_level - 1)
            continue
        
        # Manejar declaraciones de variables
        match = re.match(
            r'^\s*(int|double|float|String|boolean|char)\s+([a-zA-Z_]\w*)\s*(=\s*(.*?))?$', 
            line
        )
        if match:
            var_type, var_name, _, assignment = match.groups()
            if assignment:
                line = f"{var_name} = {assignment.strip()}"
            else:
                line = f"{var_name} = None"
        
        # Manejar incremento de indentación
        if line.endswith(':'):
            new_line = '    ' * indent_level + line
            new_lines.append(new_line)
            indent_level += 1
            continue
        
        # Manejar cierre de bloques implícito
        if in_block and not line.startswith('    ' * (indent_level - 1)):
            indent_level -= 1
            in_block = False
        
        # Aplicar indentación actual
        new_line = '    ' * indent_level + line
        
        # Manejar bloques de código
        if ':' in line:
            in_block = True
        
        new_lines.append(new_line)
    
    # Unir y limpiar líneas vacías
    python_code = '\n'.join(new_lines)
    
    # Post-procesamiento para mejor formato
    python_code = re.sub(r':\s*$', ':', python_code, flags=re.MULTILINE)  # Asegurar : al final de condicionales
    python_code = re.sub(r'print\((.*?)\)', lambda m: f'print({m.group(1).strip()})', python_code)  # Limpiar prints
    
    # Eliminar líneas vacías múltiples
    python_code = re.sub(r'\n\s*\n', '\n\n', python_code)
    
    return python_code.strip()

############################################
# FUNCIONES DE ANÁLISIS
############################################

def analizar_sintaxis(codigo_fuente):
    tokens, errores_lexicos, _ = analizador_lexico(codigo_fuente)
    parser = Parser(tokens)
    ast_root = parser.parse()
    errores = errores_lexicos + parser.errores
    return ast_root, errores, parser.codigo_intermedio, parser.symbol_table

############################################
# INTERFAZ CON CUSTOMTKINTER
############################################


class SemanticAnalyzer:
    """
    Recorre el AST y valida reglas semánticas:
    - Declaraciones duplicadas
    - Variables no definidas
    - Conflictos de tipos en operaciones y asignaciones
    - Alcance de variables (a nivel de programa en este ejemplo)
    - Conversiones inválidas (int <-> float, etc.)
    """
    def __init__(self, symbol_table, error_list, source_lines):
        # symbol_table: instancia de SymbolTable poblada en sintaxis
        # error_list: lista compartida para agregar errores semánticos
        # source_lines: líneas del código fuente para localización
        self.symbol_table = symbol_table
        self.errors = error_list
        self.lines = source_lines

    def analyze(self, node):
        # Dispatch según tipo de nodo
        method = getattr(self, f"visit_{node.nodetype}", self.generic_visit)
        return method(node)

    def generic_visit(self, node):
        for child in node.children:
            self.analyze(child)

    def visit_Print(self, node):
        # Verificar que la expresión a imprimir sea válida
        if node.children:
            expr_type = self.analyze(node.children[0])
            # Cualquier tipo puede ser impreso (en Java todo puede convertirse a String)
            return 'void'
        return 'void'

    def visit_Program(self, node):
        # Iniciar recorrido de declaraciones y sentencias
        self.generic_visit(node)

    def visit_Declaracion(self, node):
        identifier = node.value
        declared_type = self.symbol_table.get_type(identifier)
        
        # Si hay una expresión de inicialización
        if node.children:
            expr_node = node.children[0]
            expr_type = self.analyze(expr_node)
            
            # Verificar compatibilidad de tipos
            if not self.symbol_table.is_conversion_allowed(expr_type, declared_type):
                line, col = self._loc(node)
                self.errors.append((
                    f"Error de tipo en declaración: no se puede convertir {expr_type} a {declared_type}",
                    line, col
                ))
        
        return declared_type

    def visit_Asignacion(self, node):
        identifier = node.value
        # Verificar definición
        if not self.symbol_table.exists(identifier):
            line, col = self._loc(node)
            self.errors.append((
                f"Variable no declarada: '{identifier}'", line, col
            ))
            target_type = 'error'
        else:
            target_type = self.symbol_table.get_type(identifier)

        # Evaluar expresión
        if node.children:
            expr_node = node.children[0]
            expr_type = self.analyze(expr_node)
            # Chequeo de tipos
            if target_type != expr_type and target_type != 'error':
                line, col = self._loc(node)
                self.errors.append((
                    f"Error de tipo: asignación de {expr_type} a '{identifier}' ({target_type})", line, col
                ))
        return target_type

    def visit_Operacion(self, node):
        left_type = self.analyze(node.children[0])
        right_type = self.analyze(node.children[1])
        op = node.value
        
        # Operaciones aritméticas (+, -, *, /)
        if op in {'+', '-', '*', '/'}:
            # Tipos numéricos permitidos
            numeric_types = {'int', 'float', 'double'}
            if left_type in numeric_types and right_type in numeric_types:
                # Reglas de promoción de tipos
                if 'double' in (left_type, right_type):
                    return 'double'
                elif 'float' in (left_type, right_type):
                    return 'float'
                else:
                    return 'int'
            
            # Concatenación de Strings
            elif op == '+' and 'String' in (left_type, right_type):
                # Permitir String + cualquier cosa (conversión implícita a String)
                return 'String'
            
            else:
                line, col = self._loc(node)
                self.errors.append((
                    f"Operación no soportada: {left_type} {op} {right_type}",
                    line, col
                ))
                return 'error'
        
        # Operaciones de comparación (==, !=, <, >, <=, >=)
        elif op in {'==', '!=', '<', '>', '<=', '>='}:
            # Comparación entre tipos compatibles
            if (left_type == right_type or 
                self.symbol_table.is_conversion_allowed(left_type, right_type) or
                self.symbol_table.is_conversion_allowed(right_type, left_type)):
                return 'boolean'
            else:
                line, col = self._loc(node)
                self.errors.append((
                    f"Comparación inválida: {left_type} {op} {right_type}",
                    line, col
                ))
                return 'error'
        
        # Operaciones lógicas (&&, ||)
        elif op in {'&&', '||'}:
            if left_type == 'boolean' and right_type == 'boolean':
                return 'boolean'
            else:
                line, col = self._loc(node)
                self.errors.append((
                    f"Operación lógica requiere booleanos: {left_type} {op} {right_type}",
                    line, col
                ))
                return 'error'
        
        else:
            line, col = self._loc(node)
            self.errors.append((
                f"Operador no soportado: {op}",
                line, col
            ))
            return 'error'

    def visit_Literal(self, node):
        val = node.value
        # Determinar si es cadena
        if val.startswith('"'):
            return 'String'
        # Determinar si es carácter
        elif val.startswith("'"):
            return 'char'
        # Determinar si es booleano o null
        elif val in ('true', 'false'):
            return 'boolean'
        elif val == 'null':
            return 'null'
        # Determinar tipo numérico
        else:
            lower_val = val.lower()
            # Flotante (ej: 3.14f)
            if lower_val.endswith('f'):
                return 'float'
            # Double (ej: 3.14d o 3.14)
            elif lower_val.endswith('d') or '.' in val or 'e' in val:
                return 'double'
            # Entero
            else:
                return 'int'

    def visit_Identificador(self, node):
        name = node.value
        if not self.symbol_table.exists(name):
            line, col = self._loc(node)
            self.errors.append((
                f"Variable no declarada en expresión: '{name}'", line, col
            ))
            return 'error'
        return self.symbol_table.get_type(name)

    def _loc(self, node):
        # Localización: el ASTNode podría llevar atributos de línea/columna
        # Si no, retornamos (1,1) o basado en primera línea
        return (1, 1)

class CompiladorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DJ Compilador Translator")
        self.geometry("1200x800")
        self.last_ast = None
        self.java_valid = False
        self.create_widgets()
    
    def create_widgets(self):
        # Configuración principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame principal con scroll
        main_frame = ctk.CTkScrollableFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)

        # Área de entrada de código
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        
        ctk.CTkLabel(input_frame, text="Código Java", font=("Arial", 14, "bold")).pack(anchor="w")
        self.input_text = ctk.CTkTextbox(input_frame, height=200)
        self.input_text.pack(fill="x", expand=False)

        # Botones superiores
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        ctk.CTkButton(btn_frame, text="Analizar", command=self.analizar_click).grid(row=0, column=0, padx=5)
        ctk.CTkButton(btn_frame, text="Mostrar AST", command=self.mostrar_ast_click).grid(row=0, column=1, padx=5)
        ctk.CTkButton(btn_frame, text="Borrar", command=self.borrar_click).grid(row=0, column=2, padx=5)
        ctk.CTkButton(btn_frame, text="Tema", command=self.cambiar_tema_click).grid(row=0, column=3, padx=5)
        ctk.CTkButton(btn_frame, text="Cargar .java", command=self.cargar_archivo_click).grid(row=0, column=4, padx=5)

        # Panel de resultados
        results_frame = ctk.CTkFrame(main_frame)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=5)

        # Columnas de resultados
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)

        # Columna izquierda
        left_col = ctk.CTkFrame(results_frame)
        left_col.grid(row=0, column=0, sticky="nsew", padx=5)
        
        # Tokens
        tokens_frame = ctk.CTkFrame(left_col)
        tokens_frame.pack(fill="both", expand=True, pady=5)
        ctk.CTkLabel(tokens_frame, text="Tokens", font=("Arial", 12, "bold")).pack()
        self.tokens_text = ctk.CTkTextbox(tokens_frame, height=200)
        self.tokens_text.pack(fill="both", expand=True)
        
        # AST
        ast_frame = ctk.CTkFrame(left_col)
        ast_frame.pack(fill="both", expand=True, pady=5)
        ctk.CTkLabel(ast_frame, text="Árbol Sintáctico", font=("Arial", 12, "bold")).pack()
        self.ast_text = ctk.CTkTextbox(ast_frame, height=200)
        self.ast_text.pack(fill="both", expand=True)

        # Columna derecha
        right_col = ctk.CTkFrame(results_frame)
        right_col.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Código Intermedio
        codigo_frame = ctk.CTkFrame(right_col)
        codigo_frame.pack(fill="both", expand=True, pady=5)
        ctk.CTkLabel(codigo_frame, text="Código Intermedio", font=("Arial", 12, "bold")).pack()
        self.codigo_text = ctk.CTkTextbox(codigo_frame, height=200)
        self.codigo_text.pack(fill="both", expand=True)
        
        # Errores
        errores_frame = ctk.CTkFrame(right_col)
        errores_frame.pack(fill="both", expand=True, pady=5)
        ctk.CTkLabel(errores_frame, text="Errores", font=("Arial", 12, "bold")).pack()
        self.errores_text = ctk.CTkTextbox(errores_frame, height=200)
        self.errores_text.pack(fill="both", expand=True)

        # Código Python
        python_frame = ctk.CTkFrame(main_frame)
        python_frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        python_top = ctk.CTkFrame(python_frame)
        python_top.pack(fill="x", pady=3)
        ctk.CTkLabel(python_top, text="Código Python Generado", font=("Arial", 12, "bold")).pack(side="left")
        self.download_btn = ctk.CTkButton(python_top, text="Descargar .py", command=self.descargar_python_click, state="disabled")
        self.download_btn.pack(side="right", padx=5)
        
        self.python_text = ctk.CTkTextbox(python_frame, height=100)
        self.python_text.pack(fill="x")

    # ... (Los métodos restantes se mantienen igual excepto por mejoras menores)

    def analizar_click(self):
        codigo = self.input_text.get("0.0", "end").strip()
        
        # Análisis léxico
        tokens, errores_lexicos, lineas = analizador_lexico(codigo)
        
        # Mostrar tokens
        self.tokens_text.delete("0.0", ctk.END)
        self.tokens_text.insert(ctk.END, "\n".join([f"Token: {lexema}, Tipo: {tipo}" for lexema, tipo in tokens]))
        
        # Análisis sintáctico
        parser = Parser(tokens)
        ast_root = parser.parse()
        self.last_ast = ast_root
        
        # Mostrar AST
        self.ast_text.delete("0.0", ctk.END)
        self.ast_text.insert(ctk.END, str(ast_root))
        
        # Mostrar código intermedio
        self.codigo_text.delete("0.0", ctk.END)
        self.codigo_text.insert(ctk.END, "\n".join(parser.codigo_intermedio))
        
        # Análisis semántico
        analyzer = SemanticAnalyzer(parser.symbol_table, parser.errores, lineas)
        analyzer.analyze(ast_root)
        errores_total = errores_lexicos + parser.errores
        
        # Mostrar errores
        self.errores_text.delete("0.0", ctk.END)
        self.errores_text.insert(ctk.END, "\n".join(str(e) for e in errores_total))
        
        # Generar Python si no hay errores
        self.java_valid = len(errores_total) == 0
        self.download_btn.configure(state="normal" if self.java_valid else "disabled")
        
        self.python_text.delete("0.0", ctk.END)
        if self.java_valid:
            try:
                python_code = convert_java_to_python("//Compilacion Exitosa/n" + codigo)
                self.python_text.insert(ctk.END, "//Compilacion Exitosa/n" + python_code)
            except Exception as e:
                self.python_text.insert(ctk.END, f"Error en conversión: {str(e)}")
                self.download_btn.configure(state="disabled")
        else:
            self.python_text.insert(ctk.END, "Error al Compilar")
        codigo = self.input_text.get("0.0", "end").strip()
        tokens, errores_lexicos, lineas = analizador_lexico(codigo)
        
        # Mostrar tokens
        tokens_result = "\n".join([f"Token: {lexema}, Tipo: {tipo}" for lexema, tipo in tokens])
        self.tokens_text.delete("0.0", ctk.END)
        self.tokens_text.insert(ctk.END, tokens_result)
        
        # Analizar sintaxis
        ast_root, errores_sintacticos, codigo_intermedio, symbol_table = analizar_sintaxis(codigo)
        self.last_ast = ast_root
        
        # Análisis semántico
        analyzer = SemanticAnalyzer(symbol_table, errores_sintacticos, lineas)
        analyzer.analyze(ast_root)
        
        # Actualizar interfaz
        self.ast_text.delete("0.0", ctk.END)
        self.ast_text.insert(ctk.END, ast_root.__str__())
        
        self.codigo_text.delete("0.0", ctk.END)
        self.codigo_text.insert(ctk.END, "\n".join(codigo_intermedio))
        
        self.errores_text.delete("0.0", ctk.END)
        errores_total = errores_lexicos + errores_sintacticos
        self.errores_text.insert(ctk.END, "\n".join(str(e) for e in errores_total))
        
        # Actualizar estado de validación
        self.java_valid = len(errores_total) == 0
        self.download_btn.configure(state="normal" if self.java_valid else "disabled")
        
        # Generar código Python si es válido
        self.python_text.delete("0.0", ctk.END)
        if self.java_valid:
            try:
                python_code = convert_java_to_python("//Compilacion Exitosa "+ codigo)
                self.python_text.insert(ctk.END, "//Compilacion Exitosa "+python_code)
            except Exception as e:
                self.python_text.insert(ctk.END, f"Error en conversión: {str(e)}")
                self.download_btn.configure(state="disabled")
        else:
            self.python_text.insert(ctk.END, "Error En la compilacion")
        codigo = self.input_text.get("0.0", "end").strip()
        tokens, errores_lexicos, _ = analizador_lexico(codigo)
        # Mostrar tokens
        tokens_result = "\n".join([f"Token: {lexema}, Tipo: {tipo}" for lexema, tipo in tokens])
        self.tokens_text.delete("0.0", ctk.END)
        self.tokens_text.insert(ctk.END, tokens_result)
        
        # Analizar sintaxis
        ast_root, errores_sintacticos, codigo_intermedio, symbol_table = analizar_sintaxis(codigo)
        self.last_ast = ast_root
        self.ast_text.delete("0.0", ctk.END)
        self.ast_text.insert(ctk.END, ast_root.__str__())
        
        self.codigo_text.delete("0.0", ctk.END)
        self.codigo_text.insert(ctk.END, "\n".join(codigo_intermedio))
        
        self.errores_text.delete("0.0", ctk.END)
        errores_total = errores_lexicos + errores_sintacticos
        self.errores_text.insert(ctk.END, "\n".join(str(e) for e in errores_total))
    
        self.errores_text.delete("0.0", ctk.END)
        errores_total = errores_lexicos + errores_sintacticos
        self.errores_text.insert(ctk.END, "\n".join(str(e) for e in errores_total))
        
        # Actualizar estado de validación y botón
        self.java_valid = len(errores_total) == 0
        self.download_btn.configure(state="normal" if self.java_valid else "disabled")
        
        # Mostrar código Python si está válido
        self.python_text.delete("0.0", ctk.END)
        if self.java_valid:
            try:
                python_code = convert_java_to_python("//Compilacion Exitosa" + codigo)
                self.python_text.insert(ctk.END, python_code)
            except Exception as e:
                self.python_text.insert(ctk.END, f"Error en conversión: {str(e)}")
                self.download_btn.configure(state="disabled")
        else:
            self.python_text.insert(ctk.END, "Error en la compilacion")

    def descargar_python_click(self):
        if not self.java_valid:
            messagebox.showerror("Error", "El código Java tiene errores. No se puede descargar.")
            return
            
        python_code = self.python_text.get("0.0", "end").strip()
        if not python_code:
            messagebox.showerror("Error", "No hay código Python para descargar")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(python_code)
                messagebox.showinfo("Éxito", "Archivo guardado correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar:\n{str(e)}")

    def borrar_click(self):
        self.input_text.delete("0.0", ctk.END)
        self.tokens_text.delete("0.0", ctk.END)
        self.ast_text.delete("0.0", ctk.END)
        self.codigo_text.delete("0.0", ctk.END)
        self.errores_text.delete("0.0", ctk.END)
        self.python_text.delete("0.0", ctk.END)
        self.last_ast = None
        self.java_valid = False
        self.download_btn.configure(state="disabled")
        
    def mostrar_ast_click(self):
        if self.last_ast is None:
            messagebox.showerror("Error", "Primero analiza el código para generar el AST.")
            return
        try:
            graph = generar_ast_graph(self.last_ast)
            png_bytes = graph.create_png()
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar la imagen del AST: {e}")
            return
        
        # Convertir los bytes a imagen usando PIL
        image = Image.open(BytesIO(png_bytes))
        photo = ImageTk.PhotoImage(image)
        
        # Crear una ventana Toplevel para mostrar la imagen
        top = ctk.CTkToplevel(self)
        top.title("AST - Vista Gráfica")
        top.geometry("650x500")
        lbl = ctk.CTkLabel(top, image=photo, text="")
        lbl.image = photo  # Mantener referencia
        lbl.pack(padx=10, pady=10, fill="both", expand=True)
        ctk.CTkButton(top, text="Cerrar", command=top.destroy).pack(pady=10)
    
    def borrar_click(self):
        self.input_text.delete("0.0", ctk.END)
        self.tokens_text.delete("0.0", ctk.END)
        self.ast_text.delete("0.0", ctk.END)
        self.codigo_text.delete("0.0", ctk.END)
        self.errores_text.delete("0.0", ctk.END)
        self.last_ast = None
    
    def cambiar_tema_click(self):
        # Cambiar entre tema oscuro y claro usando CustomTkinter
        current_mode = ctk.get_appearance_mode()
        new_mode = "dark" if current_mode == "light" else "light"
        ctk.set_appearance_mode(new_mode)
    
    def cargar_archivo_click(self):
        file_path = filedialog.askopenfilename(filetypes=[("Java Files", "*.java")])
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    contenido = file.read()
                    self.input_text.delete("0.0", ctk.END)
                    self.input_text.insert(ctk.END, contenido)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

if __name__ == "__main__":
    app = CompiladorApp()
    app.mainloop()

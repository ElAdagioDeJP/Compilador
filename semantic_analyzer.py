# semantic_analyzer.py

from prueba import ASTNode, SymbolTable  # Ajusta 'tu_proyecto' al nombre de tu paquete principal

class SemanticAnalyzer:
    def __init__(self):
        # Usamos una tabla de símbolos independiente para la fase semántica
        self.errors = []               # Lista de tuplas (línea, mensaje)
        self.symbol_table = SymbolTable()

    def analyze(self, node, line=1):
        """ Punto de entrada: recorre todo el AST """
        method = f"analyze_{node.nodetype.lower()}"
        analyser = getattr(self, method, self.generic_analyze)
        return analyser(node, line)

    def generic_analyze(self, node, line):
        for child in node.children:
            self.analyze(child, line)

    def analyze_program(self, node, line):
        for child in node.children:
            self.analyze(child, line)

    def analyze_declaracion(self, node, line):
        ident = node.value
        var_type = "int"  # Solo implementado int por ahora
        # Si hay inicialización, verificamos tipos
        if node.children:
            expr = node.children[0]
            expr_type = self.analyze(expr, line)
            if expr_type and expr_type != var_type:
                self.errors.append((line,
                    f"Incompatibilidad de tipos: no se puede asignar {expr_type} a {var_type}"))
        # Declarar en la tabla semántica
        if not self.symbol_table.declare(ident, var_type):
            self.errors.append((line,
                f"Declaración duplicada: variable '{ident}' ya declarada"))

    def analyze_asignacion(self, node, line):
        ident = node.value
        # Verificar existencia
        if not self.symbol_table.exists(ident):
            self.errors.append((line,
                f"Variable no declarada: '{ident}'"))
            var_type = None
        else:
            var_type = self.symbol_table.get_type(ident)
        # Verificar tipo de la expresión
        if node.children:
            expr = node.children[0]
            expr_type = self.analyze(expr, line)
            if var_type and expr_type and var_type != expr_type:
                self.errors.append((line,
                    f"Incompatibilidad de tipos en asignación a '{ident}': {expr_type} ≠ {var_type}"))

    def analyze_operacion(self, node, line):
        left, right = node.children
        lt = self.analyze(left, line)
        rt = self.analyze(right, line)
        if lt and rt and lt != rt:
            self.errors.append((line,
              f"Incompatibilidad de tipos en operación '{node.value}': {lt} y {rt}"))
        # La operación hereda el tipo de sus operandos (asumimos homogeneidad)
        return lt or rt

    def analyze_literal(self, node, line):
        # Inferimos tipo a partir del lexema
        val = node.value
        return "float" if ('.' in val or 'e' in val.lower()) else "int"

    def analyze_identificador(self, node, line):
        ident = node.value
        if not self.symbol_table.exists(ident):
            self.errors.append((line,
                f"Variable no declarada: '{ident}'"))
            return None
        return self.symbol_table.get_type(ident)

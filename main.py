import flet as ft
import re

# Definición de los patrones de tokens con expresiones regulares
TOKEN_PATTERNS = [
    # Literales especiales
    (r'\b(true|false|null)\b', 'LITERAL_BOOLEANO_NULO'),
    
    # Palabras clave completas de Java (incluyendo Java 8+ y características modernas)
    (r'\b(abstract|assert|boolean|break|byte|case|catch|char|class|const|'
     r'continue|default|do|double|else|enum|extends|final|finally|float|for|'
     r'goto|if|implements|import|instanceof|int|interface|long|native|new|'
     r'package|private|protected|public|return|short|static|strictfp|super|'
     r'switch|synchronized|this|throw|throws|transient|try|var|void|volatile|'
     r'while|record|sealed|non-sealed|yield|module|open|opens|requires|'
     r'exports|provides|to|with|uses|transitive|permits)\b', 'PALABRA_CLAVE'),
    
    # Operadores y separadores actualizados
    (r'>>>=|>>=|<<=|\+\+|--|->|>=|<=|==|!=|&&|\|\||<<|>>>|>>|::|\.\.\.|'
     r'[+\-*/%&|^<>]=?|~|\?|:|@|={1,3}|!|\+|\-|\*|/|%|&|\||\^|<|>|\(|\)|\[|\]|\{|\}|;|,|\.', 'OPERADOR/SEPARADOR'),
    
    # Literales numéricos
    (r'0[xX][0-9a-fA-F_]+[lL]?', 'LITERAL_HEXADECIMAL'),
    (r'0[bB][01_]+[lL]?', 'LITERAL_BINARIO'),
    (r'0[0-7_]+[lL]?', 'LITERAL_OCTAL'),
    (r'((\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d[\d_]*)?[fFdD]?)|(\d[\d_]*[eE][+-]?\d[\d_]*[fFdD]?)|(\d[\d_]*[fFdD])', 'LITERAL_FLOTANTE'),
    (r'\d[\d_]*[lL]?', 'LITERAL_ENTERO'),
    
    # Literales de texto
    (r'"(\\\\|\\"|[^"])*"', 'LITERAL_CADENA'),
    (r"'(\\\\|\\'|[^']|\\u[0-9a-fA-F]{4})'", 'LITERAL_CARACTER'),
    
    # Identificadores
    (r'[a-zA-Z_$][a-zA-Z0-9_$]*', 'IDENTIFICADOR'),
]

# Patrones para ignorar (espacios, comentarios, etc.)
IGNORE_PATTERNS = [
    r'\s+',  # Espacios en blanco (incluye tabulaciones y saltos de línea)
    r'//.*',  # Comentarios de una línea
    r'/\*(.|\n)*?\*/',  # Comentarios multilínea
]

# Compilamos los patrones
TOKEN_REGEX = [(re.compile(pattern), token_type) for pattern, token_type in TOKEN_PATTERNS]
IGNORE_REGEX = [re.compile(pattern) for pattern in IGNORE_PATTERNS]

def analizador_lexico(codigo_fuente):
    """
    Analiza el código fuente y retorna una lista de tokens encontrados y errores.
    Ignora espacios en blanco, comentarios y otros elementos irrelevantes.
    """
    tokens = []
    errores = []
    lineas = codigo_fuente.split('\n')  # Dividir el código en líneas
    posicion = 0
    num_linea = 1  # Contador de líneas
    while posicion < len(codigo_fuente):
        # Ignorar espacios en blanco y comentarios
        ignorado = False
        for regex in IGNORE_REGEX:
            match = regex.match(codigo_fuente, posicion)
            if match:
                # Actualizar el número de línea si hay un salto de línea
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

        # Si no se encontró un token válido, es un error léxico
        if not coincidencia:
            # Calcular la línea y columna del error
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
            posicion += 1  # Avanzar para evitar bucles infinitos
    return tokens, errores, lineas

def main(page: ft.Page):
    page.title = "Analizador Léxico"
    page.theme_mode = ft.ThemeMode.LIGHT  # Modo claro por defecto

    # Campo de entrada para el código fuente
    input_code = ft.TextField(
        label="Código Java",
        multiline=True,
        expand=True,
        height=200,
        text_size=14,
        hint_text="Escribe o pega el código Java aquí..."
    )

    # Campo de salida para los resultados del análisis léxico
    output_tokens = ft.TextField(
        label="Tokens Analizados",
        multiline=True,
        expand=True,
        height=200,
        text_size=14,
        read_only=True
    )

    # Campo de salida para los errores encontrados
    output_errors = ft.TextField(
        label="Errores Encontrados",
        multiline=True,
        expand=True,
        height=200,
        text_size=14,
        read_only=True,
        color=ft.colors.RED,  # Texto en color rojo para resaltar errores
    )

    # Función para analizar el código al hacer clic en "Analizar"
    def analizar_click(e):
        codigo = input_code.value
        tokens, errores, lineas = analizador_lexico(codigo)
        resultado_tokens = "\n".join([f"Token: {token}, Tipo: {tipo}" for token, tipo in tokens])
        output_tokens.value = resultado_tokens

        # Formatear errores como lo hacen los compiladores
        resultado_errores = []
        for error, num_linea, columna, tipo_error in errores:
            linea_codigo = lineas[num_linea - 1]
            resultado_errores.append(
                f"Error en línea {num_linea}, columna {columna}: {tipo_error} '{error}'\n"
                f"Código: {linea_codigo}\n"
                f"{' ' * (columna - 1)}^\n"
            )
        output_errors.value = "\n".join(resultado_errores)
        page.update()

    # Función para borrar los campos al hacer clic en "Borrar"
    def borrar_click(e):
        input_code.value = ""
        output_tokens.value = ""
        output_errors.value = ""
        page.update()

    # Función para alternar entre modos claro y oscuro
    def cambiar_tema_click(e):
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        page.update()

    # Botones de la interfaz
    btn_analizar = ft.ElevatedButton("Analizar", on_click=analizar_click)
    btn_borrar = ft.ElevatedButton("Borrar", on_click=borrar_click)
    btn_cambiar_tema = ft.ElevatedButton("Cambiar Tema", on_click=cambiar_tema_click)

    # Agregar los elementos a la página
    page.add(
        input_code,
        ft.Row([btn_analizar, btn_borrar, btn_cambiar_tema], alignment=ft.MainAxisAlignment.CENTER),
        output_tokens,
        output_errors,
    )

ft.app(target=main)
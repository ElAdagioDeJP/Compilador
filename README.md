# Manual de Usuario - Analizador Léxico en Python
https://github.com/DearDayz/Compilador trabajo en conjunto

## 1. Introducción

Este manual describe el uso y funcionamiento del Analizador Léxico implementado en Python. Su objetivo es escanear un código fuente, identificar tokens como palabras clave, identificadores, operadores y detectar errores léxicos.

## 2. Requisitos del Sistema

Python 3.x instalado

Un editor de texto o entorno de desarrollo (VS Code, PyCharm, etc.)

## 3. Instalación y Ejecución

Guardar el código en un archivo llamado analizador_lexico.py.

Abrir una terminal en la ubicación del archivo.

## Ejecutar el comando:

python analizador_lexico.py

Se mostrará una lista de tokens identificados en el código de prueba.

## 4. Funcionamiento

El analizador léxico procesa un código fuente y devuelve una lista de tokens. Maneja los siguientes elementos:

### 4.1 Tokens Reconocidos

Palabras clave: if, else, while, for, return.

Identificadores: Variables o nombres de funciones.

Números: Enteros y decimales.

Operadores: Aritméticos (+, -, *, /) y relacionales (==, !=, <, >, <=, >=).

Asignación: =.

Puntuación: Paréntesis, llaves y punto y coma.

Comentarios: De una línea // y multilínea /* */ (se ignoran en el análisis).

Espacios en blanco: Son ignorados.

### 4.2 Manejo de Errores

Si un carácter desconocido es encontrado, se muestra un mensaje de error con su posición en el código:

Error léxico: Carácter no reconocido 'X' en la posición Y

## 5. Modificaciones y Personalización

Para agregar nuevos tokens, edite la lista TOKEN_PATTERNS en el código. Por ejemplo, para incluir cadenas de texto:

(r'\".*?\"', 'CADENA')

## 6. Ejemplo de Uso

*El siguiente código de prueba:*

int x = 10.5;
if (x > 5) {
    x = x - 1;
}

**Generará la siguiente salida:**

Token: int, Tipo: PALABRA_CLAVE
Token: x, Tipo: IDENTIFICADOR
Token: =, Tipo: ASIGNACION
Token: 10.5, Tipo: NUMERO
Token: ;, Tipo: PUNTO_Y_COMA
...

## 7. Conclusión

Este analizador léxico proporciona una base para futuras fases de compilación. Puede extenderse para reconocer más tokens o integrarse con un analizador sintáctico.

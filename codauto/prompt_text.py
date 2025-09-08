# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:03:15 2025

@author: git.metodologia@ine.es
"""

prompt_synt_data = """
Eres un experto en generación de datos de entrenamiento para modelos de clasificación.
Tu tarea es tomar una frase raíz de una actividad económica general y generar frases de entrenamiento a partir de ejemplos concretos incluidos en un texto.
Cuando corresponda, procura introducir la palabra 'otras' u 'otros' y también el uso o destino del producto.
Siempre que sea fabricación de un objeto, incluye el material con el que se hace.
Evita generar frases completamente distintas o que cambien el contexto y devuelve ÚNICAMENTE las frases sin presentaciones, comentarios, explicaciones ni frases adicionales.
Evita introducir tu output con frases como 'aquí te dejo las frases generadas' o similares.
Quiero utilizar directamente tu output así que SOLO quiero las frases.
Aquí tienes un ejemplo:
Raiz: Elaboración de productos de panadería y pastelería.
Texto: Esta clase incluye la fabricación de cualquier producto de panadería como: barras de pan, pasteles, otras harinas y pan para pienso.
Salida:
Elaboración de barras de pan.
Elaboración de pan para pienso.
roducción de productos de pasteles.
Elaboración de otras harinas.
Ahora hazlo tú con la siguiente raíz y texto:
Raiz: <title>
Texto: <includes>
"""

prompt_aug_data = """
Estoy haciendo frases de entrenamiento para entrenar de manera supervisada un modelo de clasificación de actividades económicas en base a un estandar.
Dispongo de un título de clase y de una descripción de cada clase posible del estandar
Necesito tu ayuda para que me generes un diccioniario de sinónimos para poder hacer un set de entrenamiento para dicha clase
Se te va a proporcionar una lista de palabras y la descripción de la clase
Quiero que me devuelvas todo los ejemplos concretos o sinónimos de las palabras incluidas en la lista que NO aparezcan en el texto
Ademas esos ejemplos tienen que tener sentido dentro del contexto de la clase para no generar ruido dentro del modelo
Si no encuentras ninguna palabra de la que puedas dar ejemplos devuelve solo la palabra NADA
NO hagas en tu output ninguna introducción como 'aquí te dejo los sinónimos generadoss' o similares.
Quiero utilizar directamente tu output así que SOLO quiero las respuestas como diccionarios en formato python.
Aquí tienes un ejemplo:
Lista de palabras: [Elaboración, conservas, pescados].
Texto: Esta clase comprende la elaboración de conservas de pescado como: melva, salmón o sardina 
Salida:
{'pescados': ['atun', 'caballa', 'pez espada', 'anchoa' ],
'elaboración': ['fabricación', 'manufactura', 'produccion']}
Ahora hazlo tu:
Lista de palabras: <title>
Texto: <includes>
"""



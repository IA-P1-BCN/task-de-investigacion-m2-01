# Algoritmos de Clustering

> **Tarea de investigación — Módulo 2** · Bootcamp de IA
> Autora: [carovasq](https://github.com/carovasq)

Investigación sobre los algoritmos de clustering: cómo funcionan, sus principales
tipos, ventajas, desventajas y aplicaciones en el mundo real. El repositorio incluye
además una comparativa práctica ejecutada con `scikit-learn` sobre datasets de
juguete, para contrastar la teoría con resultados reales.

## Índice

1. [¿Qué es el aprendizaje no supervisado y qué es un algoritmo de clustering?](#1-qué-es-el-aprendizaje-no-supervisado-y-qué-es-un-algoritmo-de-clustering)
2. [Clustering por centroides vs. clustering jerárquico](#2-clustering-por-centroides-vs-clustering-jerárquico)
3. [Clustering basado en densidad: DBSCAN](#3-clustering-basado-en-densidad-dbscan)
4. [Elegir el número de clusters y evaluar la calidad del agrupamiento](#4-elegir-el-número-de-clusters-y-evaluar-la-calidad-del-agrupamiento)
5. [Los cuatro algoritmos en detalle](#5-los-cuatro-algoritmos-en-detalle)
6. [Comparativa práctica](#6-comparativa-práctica)
7. [Referencias](#7-referencias)

## Estructura del repositorio

```text
.
├── README.md          <- las respuestas a las 5 preguntas (la entrega)
├── docs/              <- enunciado original de la tarea
├── img/               <- gráficos generados por el script
└── src/               <- código de la comparativa práctica
```

---

## 1. ¿Qué es el aprendizaje no supervisado y qué es un algoritmo de clustering?

El **aprendizaje no supervisado** es la rama del *machine learning* que trabaja con
datos **sin etiquetar**: el conjunto de entrenamiento contiene únicamente las
variables de entrada (`X`), sin una respuesta correcta (`y`) asociada a cada
observación. El algoritmo no aprende a predecir un valor conocido, sino a
**descubrir la estructura que ya existe en los datos**.

Dentro de esta familia conviven varias tareas —reducción de dimensionalidad (PCA),
reglas de asociación, detección de anomalías, estimación de densidad— y entre ellas
la más representativa: el clustering.

### Qué es un algoritmo de clustering

Un **algoritmo de clustering** reparte las observaciones en grupos (*clusters*) de
forma que los elementos de un mismo grupo se parezcan mucho entre sí y se
diferencien de los de los demás grupos. Es la traducción algorítmica de una idea
muy intuitiva: *«junta lo que se parece, separa lo que no»*.

Ese «parecerse» no es una opinión, sino una **medida de distancia** calculada sobre
las variables:

- **Distancia euclídea**: la línea recta entre dos puntos. Es la opción por defecto
  y la que usan K-Means o DBSCAN.
- **Distancia de Manhattan**: la suma de las diferencias en cada variable; más
  robusta frente a valores extremos.
- **Similitud del coseno**: mide el ángulo en lugar de la separación; es la
  habitual con texto, donde importa la proporción de palabras y no su cantidad.

Como todas estas medidas suman magnitudes distintas, **escalar los datos
(estandarizarlos) antes de agrupar es prácticamente obligatorio**: si una variable
va en euros (0–50.000) y otra en años (18–80), la primera dominará la distancia y
el agrupamiento acabará reflejando solo el sueldo.

### Diferencias con el aprendizaje supervisado

| | Aprendizaje supervisado | Aprendizaje no supervisado |
|---|---|---|
| **Datos de entrada** | `X` + etiquetas `y` | solo `X` |
| **Objetivo** | predecir `y` para observaciones nuevas | descubrir la estructura oculta |
| **Quién define las categorías** | un humano, al etiquetar los datos | el propio algoritmo: son el resultado |
| **Ejemplo de tarea** | clasificar un correo como spam / no spam | segmentar clientes en perfiles de compra |
| **Cómo se evalúa** | comparando predicción y verdad conocida: *accuracy*, F1, RMSE | con métricas internas de cohesión y separación: silueta, inercia |
| **Coste de preparar los datos** | alto: etiquetar a mano es caro y lento | bajo: se aprovechan los datos tal y como están |
| **Algoritmos típicos** | regresión logística, árboles de decisión, SVM, redes neuronales | K-Means, clustering jerárquico, DBSCAN, GMM, PCA |

La diferencia de fondo es **quién define las categorías**. En el aprendizaje
supervisado vienen dadas de antemano por alguien que etiquetó los datos; en el no
supervisado **las categorías son el resultado, no el punto de partida**.

Entre ambos extremos existe el **aprendizaje semi-supervisado**, que combina unos
pocos datos etiquetados con una gran masa de datos sin etiquetar: el escenario más
frecuente en proyectos reales, donde etiquetar a mano resulta caro.

### Una consecuencia importante: no hay una única respuesta correcta

Al no existir una verdad de referencia, **ningún agrupamiento es «el correcto»**.
Los mismos clientes agrupados por edad producen unos grupos, y agrupados por gasto
mensual, otros completamente distintos; ambos resultados son matemáticamente
válidos. Por eso el resultado de un clustering no es una conclusión cerrada sino
**una hipótesis que debe validarse con conocimiento del dominio**, y por eso las
métricas de calidad (ver [pregunta 4](#4-elegir-el-número-de-clusters-y-evaluar-la-calidad-del-agrupamiento))
miden la *forma* de los grupos y no su *acierto*.

### Propósito y aplicaciones reales

- **Explorar** un dataset del que aún no se sabe nada y obtener una primera
  descripción de su estructura.
- **Segmentar clientes** para campañas de marketing diferenciadas (el clásico
  análisis RFM: recencia, frecuencia e importe de compra).
- **Detectar anomalías**: lo que no encaja en ningún grupo suele ser lo valioso
  —fraude con tarjetas, intrusiones en una red, fallos en una cadena de montaje—.
- **Agrupar documentos o noticias** por tema sin haber definido antes las categorías.
- **Comprimir información**: reducir los millones de colores de una imagen a los 16
  más representativos, o resumir miles de observaciones en unos pocos perfiles.
- **Investigación biomédica**: identificar tipos celulares o subtipos de una
  enfermedad a partir de datos de expresión génica.
- **Preparar el terreno para un modelo supervisado**: el cluster asignado puede
  usarse como una variable adicional, o servir de guía para etiquetar a mano solo
  una muestra representativa.

## 2. Clustering por centroides vs. clustering jerárquico

> _Pendiente — paso 3_

## 3. Clustering basado en densidad: DBSCAN

> _Pendiente — paso 4_

## 4. Elegir el número de clusters y evaluar la calidad del agrupamiento

> _Pendiente — paso 5_

## 5. Los cuatro algoritmos en detalle

> _Pendiente — paso 6_

## 6. Comparativa práctica

> _Pendiente — pasos extra_

## 7. Referencias

> _Pendiente — paso 7_

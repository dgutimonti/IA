# Resumen del Lab 3

## ¿Qué hace el código?

El notebook implementa un flujo completo de clasificación binaria con regresión logística sobre el dataset KDD Cup 99 (intrusiones de red), desde la preparación de datos hasta la evaluación final.

## Pasos principales

1. Carga librerías para manejo de datos, cálculo numérico y visualización.
2. Lee el dataset KDD Cup 99 desde archivo local y define los nombres de columnas.
3. Convierte la etiqueta de salida a binaria:
	- `0`: tráfico normal.
	- `1`: ataque.
4. Codifica variables categóricas (`protocol_type`, `service`, `flag`) a valores numéricos con `factorize`.
5. Separa los datos en:
	- `X`: características.
	- `y`: etiqueta.
6. Mezcla aleatoriamente los datos con semilla fija (`42`) y divide en entrenamiento/prueba (80/20).
7. Normaliza las características usando media y desviación estándar del conjunto de entrenamiento, y aplica esa misma transformación al conjunto de prueba.
8. Agrega término de sesgo (columna de unos) a `X_train` y `X_test`.
9. Implementa desde cero la regresión logística:
	- Función sigmoide (con `clip` para estabilidad numérica).
	- Función de costo logístico y gradiente.
	- Descenso por gradiente para optimizar `theta`.
10. Entrena el modelo con `alpha = 0.5` y `1000` iteraciones.
11. Grafica la convergencia del costo, observando descenso sostenido y estabilización.
12. Evalúa en test con umbral `0.5`, calcula precisión y muestra ejemplos de predicción vs valor real.

## Conclusión

Se construye y evalúa un clasificador logístico sin usar modelos prearmados de scikit-learn, validando que el entrenamiento converge y que el modelo generaliza en el conjunto de prueba.
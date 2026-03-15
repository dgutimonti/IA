# Resumen del Laboratorio #1: Regresión Lineal Multivariable (ATP Matches)

### Objetivo
Desarrollar un modelo de **regresión lineal multivariable** desde cero (usando NumPy y Pandas) para predecir la **duración en minutos** de los partidos de tenis de la ATP, cumpliendo con los requisitos de un dataset con $n \ge 10$ características y $m \ge 10000$ registros.

### Dataset Utilizado
Se seleccionó el dataset `atp_matches_till_2022.csv` de la carpeta `Datasets/1_ATP_matches_D/`, el cual contiene información detallada de partidos profesionales.
- **Registros iniciales:** >180,000.
- **Registros tras limpieza:** >80,000 (filtrando valores nulos en columnas críticas).
- **Características ($n$):** 15 variables numéricas seleccionadas (edad, altura, aces, puntos de servicio, break points salvados, etc.).

### Paso a Paso del Desarrollo

**1. Preprocesamiento con Pandas**
- Carga del archivo CSV y selección de columnas numéricas relevantes para la duración del partido.
- Eliminación de filas con valores faltantes (`dropna`) para asegurar la integridad del entrenamiento.
- Separación de la variable objetivo (`minutes`) de las características.

**2. Normalización de Características**
- Implementación de la función `featureNormalize` para aplicar escalado (Z-score normalization): $x_{norm} = \frac{x - \mu}{\sigma}$.
- Esto es crítico para que el descenso por gradiente converja de manera eficiente dado que las escalas de "edad" y "puntos de servicio" son muy distintas.

**3. Implementación del Modelo (Desde Cero)**
- **Función de Costo ($J$):** Implementación del Error Cuadrático Medio (MSE) para evaluar la precisión de los parámetros $	heta$.
- **Descenso por Gradiente:** Algoritmo iterativo para actualizar los pesos $	heta$ minimizando la función de costo.
- **Vectorización:** Uso de operaciones de matrices con NumPy para optimizar el rendimiento del cálculo.

**4. Entrenamiento y Optimización**
- Configuración de hiperparámetros: $\alpha = 0.01$ e iteraciones = 1000.
- Seguimiento del historial de costo para verificar la convergencia visualmente mediante una gráfica.

**5. Inferencia**
- Creación de un flujo de inferencia que normaliza nuevos datos (usando $\mu$ y $\sigma$ del entrenamiento) antes de realizar la predicción.
- Verificación de resultados comparando la predicción con el promedio real del dataset.

### Conclusiones
El modelo logra aproximarse a la duración real de los partidos mediante el aprendizaje de las estadísticas de juego. La implementación manual permite entender profundamente el funcionamiento del descenso por gradiente y la importancia de la normalización en problemas multivariables.

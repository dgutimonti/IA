

 

Retrieving Notebook summary.

Read [](file:///mnt/disk/Ing.%20Sistemas/Septimo%20Semestre/1.%20IA/Lab/Lab_2.ipynb#1-1), lines 1 to 299

## Resumen del Laboratorio 2 — Regresión Logística

### Objetivo
Clasificación binaria sobre el dataset **KDD Cup 1999**: predecir si una conexión de red es **normal (0)** o un **ataque (1)** en un entorno militar simulado.

### Pipeline implementado (todo desde cero con NumPy)

| Etapa | Qué se hizo |
|---|---|
| **1. Carga y exploración** | Dataset con ~494K conexiones y 41 columnas. Se exploró la distribución de labels. |
| **2. Binarización del target** | Se convirtió `label` en binario: `normal. → 0`, cualquier otro tipo de ataque → `1`. |
| **3. Selección de features** | 38 numéricas + 3 categóricas (`protocol_type`, `service`, `flag`). |
| **4. Limpieza** | Verificación y eliminación de NaN. |
| **5. One-Hot Encoding** | `pd.get_dummies` con `drop_first=True` para las 3 categóricas. |
| **6. Normalización** | Z-score manual ($\frac{x - \mu}{\sigma}$) solo sobre las columnas numéricas, preservando las dummies. |
| **7. Bias column** | Se agregó $x_0 = 1$ al frente de la matriz de features. |
| **8. Train/Test split** | 80/20 estratificado (`stratify=y`) con `random_state=42`. |
| **9. Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ — implementada y graficada. |
| **10. Hipótesis** | $h_\theta(x) = \sigma(\theta^T x)$ |
| **11. Función de costo** | Log-loss: $J(\theta) = -\frac{1}{m}\sum\left[y\log(h) + (1-y)\log(1-h)\right]$ con `np.clip` para estabilidad numérica. |
| **12. Gradient Descent** | Manual, 1000 iteraciones con $\alpha = 0.1$. Se graficó la curva de convergencia de $J(\theta)$. |
| **13. Evaluación en test** | Accuracy, Precision, Recall, F1, AUC-ROC y Matriz de Confusión. |

### En resumen
Se construyó un clasificador de regresión logística **desde cero** (sin `sklearn.LogisticRegression`) para detectar intrusiones de red: desde la preparación de datos hasta la evaluación con métricas estándar. Las únicas herramientas de sklearn usadas fueron `train_test_split` para el split y `sklearn.metrics` para las métricas finales.
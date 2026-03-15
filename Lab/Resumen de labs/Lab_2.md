

 

Read [](file:///mnt/disk/Ing.%20Sistemas/Septimo%20Semestre/1.%20IA/Lab/Lab_1.ipynb)

Acá va el resumen de todo lo que se hizo en el cuadernillo:

---

## Resumen del Laboratorio #1

### Objetivo
Predecir la **duración en minutos** de partidos de tenis ATP usando **regresión lineal multivariable implementada desde cero** (sin sklearn para el modelo, solo NumPy).

### Paso a paso

**1. Carga y exploración del dataset**
- Se carga `atp_matches_till_2022.csv` (partidos de tenis ATP hasta 2022)
- Se exploran dimensiones, estadísticas descriptivas y valores faltantes (`NaN`)

**2. Selección de features y limpieza**
- Se eligen **21 variables numéricas** (ranking, edad, aces, doble faltas, puntos de servicio, break points, etc.) y **3 categóricas** (surface, tourney_level, round)
- Variable objetivo (`target`): `minutes`
- Se eliminan filas con datos faltantes (`dropna`)

**3. One-Hot Encoding**
- Las 3 columnas de texto se convierten a columnas binarias (0/1) con `pd.get_dummies`, generando **16 columnas one-hot**
- Total de features: 21 numéricas + 16 one-hot = **37**

**4. Normalización (feature scaling)**
- Se normalizan **solo las 21 numéricas** con z-score: $x_{norm} = \frac{x - \mu}{\sigma}$
- Las columnas one-hot (ya son 0/1) se dejan sin tocar
- Se guardan `mu` y `sigma` para reutilizar en inferencia

**5. Preparación de matrices**
- Se agrega columna $x_0 = 1$ (bias/intercepto) → `x_b` con **38 columnas**
- Se divide 80/20 en `x_train`/`x_test` con `train_test_split`

**6. Modelo — 3 funciones implementadas a mano:**

| Función | Qué hace | Fórmula |
|---|---|---|
| `hipotesis` | Predicción | $h_\theta(x) = X_b \cdot \theta$ |
| `costo` | Mide el error | $J(\theta) = \frac{1}{2m}\sum(h_\theta(x) - y)^2$ |
| `gradient_descent` | Aprende $\theta$ | $\theta := \theta - \frac{\alpha}{m} X_b^T(X_b\theta - y)$ |

**7. Entrenamiento**
- Se ejecuta gradient descent con $\alpha = 0.01$ y 1000 iteraciones
- Se grafica la convergencia de $J(\theta)$ para verificar que el costo baja

**8. Inferencia**
- Función `inferencia` que recibe un partido nuevo, normaliza las numéricas con los mismos `mu`/`sigma`, concatena las one-hot, y aplica la hipótesis
- Se prueba con un partido inventado (Grand Slam, clay, QF) → **125.1 minutos**

**9. Evaluación**
- Se calculan métricas en el test set: **RMSE**, **MAE**, **R²**
- Se grafica predicciones vs reales (scatter plot con línea diagonal de predicción perfecta)
- Se compara contra `sklearn.LinearRegression` para validar que la implementación manual es correcta
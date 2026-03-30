# Lab 5 — Resumen detallado (PyTorch)

Este resumen explica en detalle lo que hacen los tres laboratorios Lab_5.x y las **secciones obligatorias** basadas en los notebooks de PyTorch:

- `02 pytorch_nn.ipynb` → redes neuronales (`nn`)
- `03 pytorch_datasets.ipynb` → datasets y dataloaders
- `04 pytorch_save.ipynb` → guardado/exportacion de modelos

---

# Secciones obligatorias (al inicio de cada Lab_5.x)

## Modelos secuenciales
- Uso de `nn.Sequential` para apilar capas.

## Optimizadores y funciones de perdida
- Flujo: forward → loss → backward → step.

## Modelos custom
- Hereda de `nn.Module`, define `forward`.

## Accediendo a las capas
- Inspeccion de `weight` y `bias`.

## Iterando tensores y batches
- Iteracion manual para entender batching.

## Dataset y DataLoader
- Dataset custom + DataLoader para batches.

## Guardando y exportando modelos
- `state_dict()` + TorchScript + ONNX.

---

# Fases comunes en todos los Lab_5.x

1. **Fase 1: Entender el dataset**
   - Carga, limpieza, inspeccion de datos.

2. **Fase 2: Utilizacion y entrenamiento**
   - Normalizacion, DataLoader, modelo, entrenamiento.
   - **Graficos:** curvas de loss y accuracy (si aplica).

3. **Fase 3: Pruebas**
   - Evaluacion en test.
   - Predicciones de ejemplo vs valor real.

---

## Lab_5.1 — Regresion lineal univariable (ATP)
- Dataset ATP (`w_svpt` → `minutes`).
- Grafico de perdida MSE.
- Prediccion ejemplo promedio.

## Lab_5.2 — Regresion logistica binaria (KDD Cup 99)
- Etiqueta binaria normal/ataque.
- Graficos de loss y accuracy.
- Accuracy en test y ejemplo de predicciones.

## Lab_5.3 — Clasificacion CIFAR-10
- Regresion logistica multiclase.
- Graficos de loss y accuracy.
- Accuracy en test y ejemplo de predicciones.

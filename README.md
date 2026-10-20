# Examen práctico de Deep Learning: caso RemesaSegura

Se resuelve el caso de RemesaSegura S.A., una empresa de remesas entre Guatemala y Estados Unidos que evalúa lanzar un sistema de detección de lavado de dinero. El grupo actúa como comité técnico y recomienda lanzar, lanzar con condiciones o no lanzar.

- **Task 1** (`notebooks/01` a `07`). Un notebook por rol (modelado, transferencia, series de tiempo, modelos generativos, lenguaje natural, explicabilidad y operaciones), cada uno con sus cálculos y su sección del memo.
- **Task 2** (`notebooks/08_integracion.ipynb`). Conciliación de cifras, tabla de umbrales a prevalencia real, decisión con condiciones, fallos de la IA y memo final.
- **Task 3** (`presentacion/Esquema.png`). Esquema de una página para la presentación.

## Estructura

```
├── notebooks/       Notebooks por rol (01 a 07) y de integración (08)
├── presentacion/    Esquema de la presentación
├── src/
│   ├── case_data.py   Tablas y parámetros del enunciado
│   ├── metrics.py     Métricas bajo una prevalencia dada
│   └── operations.py  Capacidad, costos y tabla de umbrales
└── requirements.txt
```

## Cómo ejecutar

Los notebooks solo realizan cálculos sobre tablas pequeñas y se ejecutan de forma local en pocos segundos.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

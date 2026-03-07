# Detección de Zonas de Revalorización Inmobiliaria

Modelo de scoring de zonas urbanas basado en un **Graph Neural Network (GNN) simplificado**. Trata cada zona como un nodo en un grafo de adyacencia geográfica y propaga información entre vecinos mediante *message-passing* para capturar efectos espaciales de gentrificación y revalorización.

La arquitectura es **independiente de la ciudad**: se adapta a cualquier mercado sustituyendo el dataset de zonas con las coordenadas y features correspondientes.

---

## Arquitectura del Modelo

```
Features por nodo (zona urbana)
  ├── tend_1a        → tendencia precio interanual (%) — peso 30%
  ├── infra          → índice inversión infraestructura (0-100) — peso 25%
  ├── licencias      → actividad nuevas licencias (0-100) — peso 20%
  ├── transporte     → acceso a transporte público — peso 15%
  └── tend_3a        → tendencia precio 3 años (%) — peso 10%

GNN Simplificado (2 rounds de message-passing)
  ├── Round 1: score_r1[i] = 0.65 × score_local[i] + 0.35 × mean(vecinos)
  └── Round 2: score_r2[i] = 0.72 × score_r1[i] + 0.28 × mean(vecinos)

Score final → normalizado a escala 0-100
```

### Grafo

- **N nodos** — zonas urbanas con features socioeconómicas (demo: 23 zonas)
- **Aristas** — conexiones de proximidad geográfica (radio configurable, ~1.5km por defecto)
- **2 rounds** de message-passing espacial

### Categorías de Score

| Score | Categoría |
|-------|-----------|
| ≥ 72  | Alta oportunidad |
| ≥ 52  | Oportunidad moderada |
| ≥ 35  | Media |
| < 35  | Baja |

---

## Estructura del proyecto

```
proyecto-revalorizacion/
├── api.py        # FastAPI standalone (puerto 8090)
├── router.py     # Endpoints REST (/ml/revalorizacion/*)
└── data.py       # Dataset de zonas + modelo GNN
```

## Endpoints

```
GET /ml/revalorizacion/mapa          → todas las zonas con score y coordenadas
GET /ml/revalorizacion/barrio/{id}   → análisis detallado de una zona
GET /ml/revalorizacion/stats         → metadatos del modelo GNN
```

## Arranque local

```bash
uvicorn api:app --host 127.0.0.1 --port 8090 --reload
```

Requiere Python 3.10+ con FastAPI y NumPy instalados.

## Stack

- **Python 3.12** — lógica del modelo y API
- **FastAPI** — servidor REST
- **NumPy** — cómputo vectorial del GNN
- **Leaflet.js** — mapa interactivo (frontend, en repositorio del portfolio)
- **CartoDB Dark** — tiles del mapa

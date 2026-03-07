# Detección de Zonas de Revalorización Inmobiliaria

Modelo de scoring de zonas urbanas basado en un **Graph Neural Network (GNN) simplificado** aplicado al mercado inmobiliario de Madrid. Trata cada barrio como un nodo en un grafo de adyacencia geográfica y propaga información entre vecinos mediante *message-passing* para capturar efectos espaciales de gentrificación y revalorización.

**Demo en vivo:** [portfolio]

---

## Arquitectura del Modelo

```
Features por nodo (barrio)
  ├── tend_1a        → tendencia precio interanual (%) — peso 30%
  ├── infra          → índice inversión infraestructura (0-100) — peso 25%
  ├── licencias      → actividad nuevas licencias (0-100) — peso 20%
  ├── metro          → líneas de metro a <500m — peso 15%
  └── tend_3a        → tendencia precio 3 años (%) — peso 10%

GNN Simplificado (2 rounds de message-passing)
  ├── Round 1: score_r1[i] = 0.65 × score_local[i] + 0.35 × mean(vecinos)
  └── Round 2: score_r2[i] = 0.72 × score_r1[i] + 0.28 × mean(vecinos)

Score final → normalizado a escala 0-100
```

### Grafo

- **23 nodos** — barrios de Madrid con features socioeconómicas
- **40 aristas** — conexiones de proximidad geográfica (~1.5km)
- **2 rounds** de message-passing espacial

### Categorías de Score

| Score | Categoría |
|-------|-----------|
| ≥ 72  | Alta oportunidad |
| ≥ 52  | Oportunidad moderada |
| ≥ 35  | Media |
| < 35  | Baja |

### Top barrios detectados

| Barrio | Score | Señales clave |
|--------|-------|---------------|
| Tetuán | 90.0 | Infraestructura 88/100, licencias 85/100, +9.8% interanual |
| Vallecas | 80.5 | +11.5% interanual, tendencia sostenida 3 años |
| Fuencarral | 80.2 | Alta actividad promotora, buena conectividad |
| Carabanchel | 77.2 | +10.2% interanual, infraestructura 82/100 |

---

## Estructura del proyecto

```
proyecto-revalorizacion/
├── api.py        # FastAPI standalone (puerto 8090)
├── router.py     # Endpoints REST (/ml/revalorizacion/*)
└── data.py       # Dataset de barrios + modelo GNN
```

## Endpoints

```
GET /ml/revalorizacion/mapa          → todos los barrios con score y coordenadas
GET /ml/revalorizacion/barrio/{id}   → análisis detallado de un barrio
GET /ml/revalorizacion/stats         → metadatos del modelo GNN
```

## Arranque local

```bash
# Requiere el venv del proyecto chatbot o cualquier entorno con FastAPI
uvicorn api:app --host 127.0.0.1 --port 8090 --reload
```

## Stack

- **Python 3.12** — lógica del modelo y API
- **FastAPI** — servidor REST
- **NumPy** — cómputo vectorial del GNN
- **Leaflet.js** — mapa interactivo (frontend, en repositorio del portfolio)
- **CartoDB Dark** — tiles del mapa

---

> Proyecto de portfolio — datos sintéticos realistas basados en fuentes públicas (Idealista, INE, Ayuntamiento de Madrid).

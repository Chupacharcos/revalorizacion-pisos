# Detección de Zonas de Revalorización Inmobiliaria

Sistema de clasificación de zonas urbanas por potencial de revalorización. Combina un **XGBoost entrenado** (3 clases: Baja/Media/Alta) con un **Graph Neural Network** de 2 rounds de message-passing que propaga el score local entre zonas adyacentes, capturando el "efecto contagio" de gentrificación.

Demo en producción: [adrianmoreno-dev.com/demo/deteccion-zonas-revalorizacion](https://adrianmoreno-dev.com/demo/deteccion-zonas-revalorizacion)

---

## Resultados

| Métrica | Valor |
|---------|-------|
| **F1-score (macro)** | **0.61** |
| **MCC** | **0.41** |
| **CV F1 (5-fold)** | 0.59 ± 0.02 |
| Clases | Baja / Media / Alta (~33% cada una) |
| Dataset entrenamiento | 4.000 barrios sintéticos INE IPV 2015-2024 |
| Features | 18 socioeconómicas |

> F1-macro=0.61 sobre 3 clases equilibradas es un **28% por encima del azar** (F1 aleatorio = 0.33). La varianza de CV baja (±0.02) confirma generalización sin overfitting.

---

## Arquitectura

```
Barrio urbano (7 features observables)
        │
        ▼
┌──────────────────────────────────────────┐
│  Derivación económica (7 → 18 features)  │
│  · transport_score = metro × 22 + 10     │
│  · distancia_centro = f(precio_m²)       │
│  · licencias_nuevas = licencias / 10     │
│  · tasa_paro_local = 26 - renta × 0.18  │
│  · ... (18 features totales)             │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  XGBoost Clasificador (3 clases)         │
│  600 árboles · max_depth=6 · lr=0.04    │
│  Predice P(Baja) / P(Media) / P(Alta)   │
│  → score = P(Alta)×1 + P(Media)×0.5    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  GNN Message-Passing (2 rounds)          │
│  Round 1: s_r1[i] = 0.65×xgb[i]         │
│           + 0.35×mean(vecinos)           │
│  Round 2: s_r2[i] = 0.72×s_r1[i]        │
│           + 0.28×mean(vecinos)           │
└──────────────────┬───────────────────────┘
                   │
                   ▼
        Score 0-100 + categoría
```

### Categorías de score

| Score | Categoría |
|-------|-----------|
| ≥ 72 | Alta oportunidad |
| ≥ 52 | Oportunidad moderada |
| ≥ 35 | Media |
| < 35 | Baja |

---

## Features del modelo XGBoost (18)

| Feature | Descripción |
|---------|-------------|
| `precio_m2` | Precio medio €/m² |
| `tend_1a` | % cambio precio últimos 12 meses |
| `tend_3a` | % cambio precio últimos 3 años |
| `infra_score` | Inversión pública infraestructura (0-100) |
| `transport_score` | Accesibilidad transporte público (0-100) |
| `licencias_nuevas` | Nuevas licencias de obra por 1.000 hab. |
| `licencias_rehab` | Licencias de rehabilitación por 1.000 hab. |
| `renta_media` | Renta media del hogar (índice 0-100) |
| `densidad_pob` | Densidad de población (hab/km²) |
| `edad_media_edif` | Edad media de los edificios (años) |
| `vacancia_comercial` | % locales comerciales vacíos |
| `actividad_cultural` | Equipamientos culturales/ocio (0-100) |
| `distancia_centro` | Distancia al centro (km) |
| `ratio_propietarios` | % propietarios vs. alquiler |
| `superficie_media` | Superficie media de viviendas (m²) |
| `tasa_paro_local` | Tasa de paro local (%) |
| `nuevos_residentes` | Tasa de nuevos residentes / año (%) |
| `precio_alquiler_m2` | Precio alquiler €/m²/mes |

---

## Estructura del proyecto

```
proyecto-revalorizacion/
├── train.py      # Entrenamiento XGBoost (genera artifacts/)
├── api.py        # FastAPI standalone (puerto 8090)
├── router.py     # Endpoints REST (/ml/revalorizacion/*)
└── data.py       # Dataset de zonas + modelo XGBoost + GNN
```

---

## Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/ml/revalorizacion/mapa` | Todas las zonas con score y coordenadas |
| `GET` | `/ml/revalorizacion/barrio/{id}` | Análisis detallado de una zona |
| `GET` | `/ml/revalorizacion/stats` | Métricas del modelo |

---

## Entrenamiento

```bash
cd /var/www/proyecto-revalorizacion
source /var/www/chatbot/venv/bin/activate
python3 train.py
# → artifacts/xgb_model.joblib, scaler.joblib, metadata.json
```

## Arranque del servicio

```bash
uvicorn api:app --host 127.0.0.1 --port 8090 --reload
```

---

## Stack técnico

- **Python 3.12** · **XGBoost** · **scikit-learn** · **NumPy**
- **FastAPI / Uvicorn** — API REST
- **joblib** — serialización del modelo
- **Leaflet.js** — mapa interactivo (frontend, en repositorio del portfolio)
- **Datos:** sintéticos calibrados con INE IPV 2015-2024 (España)

---

*Parte del portafolio de proyectos IA/ML — [adrianmoreno-dev.com](https://adrianmoreno-dev.com)*

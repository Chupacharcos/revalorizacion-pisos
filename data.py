"""
Modelo de scoring de zonas de revalorizacion para Madrid.

Arquitectura: Graph Neural Network simplificado (2 rounds de message-passing).
  - Nodos:   barrios de Madrid con features socioeconomicas
  - Aristas: proximidad geografica (adyacencia)
  - Score:   potencial de revalorizacion 0-100 en horizonte 12-24 meses

El GNN simplificado aplica:
  1. Normalizacion de features por rango global
  2. Computo del score local (combinacion lineal ponderada)
  3. Round 1 de message-passing: promedio ponderado self + vecinos
  4. Round 2 de message-passing: refinamiento espacial
  5. Normalizacion final al rango 0-100
"""

import math
from typing import Dict, List

# ── Dataset de barrios (datos sinteticos realistas a 2025) ────────────────────
# Fuentes de referencia: Idealista, INE, Ayuntamiento de Madrid
# Features:
#   precio_m2:    precio actual €/m² (referencia mercado libre 2025)
#   tend_1a:      variacion anual precio (%)
#   tend_3a:      variacion acumulada 3 anos (%)
#   infra:        indice de inversion en infraestructura (0-100)
#   metro:        numero de lineas de metro a menos de 500m
#   licencias:    actividad de nuevas licencias de construccion (0-100)
#   renta:        indice de renta media (50 = media Madrid)

BARRIOS: List[Dict] = [
    {"id": "malasana",      "nombre": "Malasana",       "lat": 40.4268, "lng": -3.7035, "precio_m2": 5200, "tend_1a": 8.2,  "tend_3a": 22.1, "infra": 72, "metro": 3, "licencias": 68, "renta": 62},
    {"id": "lavapies",      "nombre": "Lavapies",        "lat": 40.4082, "lng": -3.7035, "precio_m2": 3900, "tend_1a": 9.1,  "tend_3a": 25.3, "infra": 65, "metro": 2, "licencias": 71, "renta": 38},
    {"id": "chueca",        "nombre": "Chueca",          "lat": 40.4235, "lng": -3.6958, "precio_m2": 5400, "tend_1a": 6.8,  "tend_3a": 18.2, "infra": 55, "metro": 3, "licencias": 45, "renta": 70},
    {"id": "salamanca",     "nombre": "Salamanca",       "lat": 40.4262, "lng": -3.6820, "precio_m2": 7200, "tend_1a": 4.1,  "tend_3a": 12.3, "infra": 48, "metro": 3, "licencias": 30, "renta": 90},
    {"id": "chamberi",      "nombre": "Chamberi",        "lat": 40.4345, "lng": -3.6982, "precio_m2": 6100, "tend_1a": 5.2,  "tend_3a": 14.8, "infra": 52, "metro": 4, "licencias": 35, "renta": 80},
    {"id": "retiro",        "nombre": "Retiro",          "lat": 40.4082, "lng": -3.6820, "precio_m2": 6300, "tend_1a": 4.8,  "tend_3a": 13.1, "infra": 58, "metro": 2, "licencias": 28, "renta": 85},
    {"id": "arguelles",     "nombre": "Arguelles",       "lat": 40.4284, "lng": -3.7120, "precio_m2": 5600, "tend_1a": 5.9,  "tend_3a": 15.4, "infra": 61, "metro": 3, "licencias": 40, "renta": 72},
    {"id": "arganzuela",    "nombre": "Arganzuela",      "lat": 40.3990, "lng": -3.6940, "precio_m2": 4200, "tend_1a": 7.3,  "tend_3a": 19.8, "infra": 78, "metro": 2, "licencias": 62, "renta": 52},
    {"id": "carabanchel",   "nombre": "Carabanchel",     "lat": 40.3880, "lng": -3.7300, "precio_m2": 2800, "tend_1a": 10.2, "tend_3a": 28.7, "infra": 82, "metro": 3, "licencias": 80, "renta": 35},
    {"id": "vallecas",      "nombre": "Vallecas",        "lat": 40.3850, "lng": -3.6650, "precio_m2": 2400, "tend_1a": 11.5, "tend_3a": 31.2, "infra": 75, "metro": 2, "licencias": 78, "renta": 28},
    {"id": "tetuan",        "nombre": "Tetuan",          "lat": 40.4520, "lng": -3.6980, "precio_m2": 4100, "tend_1a": 9.8,  "tend_3a": 26.4, "infra": 88, "metro": 3, "licencias": 85, "renta": 45},
    {"id": "fuencarral",    "nombre": "Fuencarral",      "lat": 40.4800, "lng": -3.6930, "precio_m2": 3600, "tend_1a": 8.1,  "tend_3a": 21.3, "infra": 70, "metro": 2, "licencias": 72, "renta": 55},
    {"id": "hortaleza",     "nombre": "Hortaleza",       "lat": 40.4730, "lng": -3.6600, "precio_m2": 3800, "tend_1a": 7.2,  "tend_3a": 19.1, "infra": 65, "metro": 1, "licencias": 65, "renta": 58},
    {"id": "ciudad_lineal", "nombre": "Ciudad Lineal",   "lat": 40.4400, "lng": -3.6590, "precio_m2": 3500, "tend_1a": 6.8,  "tend_3a": 17.2, "infra": 60, "metro": 3, "licencias": 55, "renta": 50},
    {"id": "san_blas",      "nombre": "San Blas",        "lat": 40.4300, "lng": -3.6300, "precio_m2": 2900, "tend_1a": 5.9,  "tend_3a": 15.8, "infra": 52, "metro": 2, "licencias": 48, "renta": 40},
    {"id": "moratalaz",     "nombre": "Moratalaz",       "lat": 40.4050, "lng": -3.6510, "precio_m2": 2700, "tend_1a": 5.4,  "tend_3a": 14.2, "infra": 45, "metro": 1, "licencias": 40, "renta": 42},
    {"id": "vicalvaro",     "nombre": "Vicalvaro",       "lat": 40.4030, "lng": -3.6170, "precio_m2": 2300, "tend_1a": 7.1,  "tend_3a": 18.9, "infra": 68, "metro": 1, "licencias": 70, "renta": 32},
    {"id": "usera",         "nombre": "Usera",           "lat": 40.3900, "lng": -3.7050, "precio_m2": 2600, "tend_1a": 6.2,  "tend_3a": 16.5, "infra": 55, "metro": 2, "licencias": 52, "renta": 33},
    {"id": "latina",        "nombre": "Latina",          "lat": 40.4040, "lng": -3.7260, "precio_m2": 2900, "tend_1a": 5.8,  "tend_3a": 15.3, "infra": 50, "metro": 3, "licencias": 45, "renta": 38},
    {"id": "villaverde",    "nombre": "Villaverde",      "lat": 40.3600, "lng": -3.7000, "precio_m2": 1900, "tend_1a": 4.5,  "tend_3a": 12.1, "infra": 38, "metro": 1, "licencias": 35, "renta": 22},
    {"id": "barajas",       "nombre": "Barajas",         "lat": 40.4740, "lng": -3.5930, "precio_m2": 2800, "tend_1a": 4.2,  "tend_3a": 11.3, "infra": 42, "metro": 1, "licencias": 32, "renta": 45},
    {"id": "palacio",       "nombre": "Palacio",         "lat": 40.4143, "lng": -3.7110, "precio_m2": 5800, "tend_1a": 5.5,  "tend_3a": 14.8, "infra": 62, "metro": 2, "licencias": 38, "renta": 68},
    {"id": "moncloa",       "nombre": "Moncloa",         "lat": 40.4350, "lng": -3.7200, "precio_m2": 5100, "tend_1a": 5.1,  "tend_3a": 13.5, "infra": 55, "metro": 2, "licencias": 36, "renta": 72},
]

# Grafo de adyacencia geografica: aristas no dirigidas
ADJACENCY: Dict[str, List[str]] = {
    "malasana":      ["chueca", "chamberi", "palacio", "arguelles"],
    "lavapies":      ["palacio", "arganzuela", "retiro", "usera"],
    "chueca":        ["malasana", "chamberi", "salamanca"],
    "salamanca":     ["chueca", "chamberi", "retiro", "ciudad_lineal"],
    "chamberi":      ["malasana", "chueca", "tetuan", "arguelles", "salamanca"],
    "retiro":        ["salamanca", "lavapies", "arganzuela", "moratalaz"],
    "arguelles":     ["malasana", "chamberi", "moncloa", "palacio"],
    "arganzuela":    ["lavapies", "retiro", "usera", "vallecas", "carabanchel"],
    "carabanchel":   ["arganzuela", "latina", "usera", "villaverde"],
    "vallecas":      ["arganzuela", "moratalaz", "vicalvaro", "usera"],
    "tetuan":        ["chamberi", "fuencarral", "ciudad_lineal"],
    "fuencarral":    ["tetuan", "hortaleza"],
    "hortaleza":     ["fuencarral", "ciudad_lineal", "san_blas", "barajas"],
    "ciudad_lineal": ["tetuan", "hortaleza", "san_blas", "salamanca"],
    "san_blas":      ["ciudad_lineal", "hortaleza", "moratalaz", "vicalvaro"],
    "moratalaz":     ["retiro", "san_blas", "vallecas", "vicalvaro"],
    "vicalvaro":     ["moratalaz", "vallecas", "san_blas"],
    "usera":         ["lavapies", "arganzuela", "carabanchel", "villaverde"],
    "latina":        ["palacio", "carabanchel", "arguelles"],
    "villaverde":    ["carabanchel", "usera"],
    "barajas":       ["hortaleza"],
    "palacio":       ["malasana", "lavapies", "latina", "arguelles"],
    "moncloa":       ["arguelles", "chamberi"],
}

# Pesos del modelo de scoring local
# Fundamento: tendencia a corto plazo + actividad inversora + accesibilidad
WEIGHTS = {
    "tend_1a":   0.30,   # tendencia precio 1 ano (predictor principal)
    "infra":     0.25,   # inversion en infraestructura
    "licencias": 0.20,   # actividad nuevas construcciones
    "metro":     0.15,   # acceso a transporte publico
    "tend_3a":   0.10,   # tendencia a largo plazo (confirmacion)
}

# Rangos globales para normalizacion min-max
_RANGES = {
    "tend_1a":   (4.0, 12.0),
    "tend_3a":   (11.0, 32.0),
    "infra":     (35.0, 90.0),
    "metro":     (0.0, 4.0),
    "licencias": (30.0, 90.0),
}


def _normalize(value: float, feat: str) -> float:
    lo, hi = _RANGES[feat]
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _local_score(barrio: Dict) -> float:
    """Computa el score local (pre message-passing) en rango 0-1."""
    return (
        WEIGHTS["tend_1a"]   * _normalize(barrio["tend_1a"],   "tend_1a")
        + WEIGHTS["infra"]   * _normalize(barrio["infra"],     "infra")
        + WEIGHTS["licencias"]* _normalize(barrio["licencias"],"licencias")
        + WEIGHTS["metro"]   * _normalize(barrio["metro"],     "metro")
        + WEIGHTS["tend_3a"] * _normalize(barrio["tend_3a"],   "tend_3a")
    )


def _message_pass(scores: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Un round de message-passing: score_new = alpha*self + (1-alpha)*mean(neighbors)."""
    new_scores = {}
    for bid, score in scores.items():
        neighbors = ADJACENCY.get(bid, [])
        if neighbors:
            neighbor_mean = sum(scores[n] for n in neighbors if n in scores) / len(neighbors)
            new_scores[bid] = alpha * score + (1 - alpha) * neighbor_mean
        else:
            new_scores[bid] = score
    return new_scores


def _get_signals(barrio: Dict) -> List[Dict]:
    """Genera las senales detectadas que justifican el score."""
    signals = []

    if barrio["tend_1a"] >= 9.0:
        signals.append({"tipo": "positivo", "texto": f"Revalorizacion anual alta: +{barrio['tend_1a']}%"})
    elif barrio["tend_1a"] >= 6.0:
        signals.append({"tipo": "positivo", "texto": f"Revalorizacion anual moderada: +{barrio['tend_1a']}%"})
    else:
        signals.append({"tipo": "neutro", "texto": f"Revalorizacion anual limitada: +{barrio['tend_1a']}%"})

    if barrio["infra"] >= 75:
        signals.append({"tipo": "positivo", "texto": f"Alta inversion en infraestructura (indice {barrio['infra']}/100)"})
    elif barrio["infra"] >= 55:
        signals.append({"tipo": "neutro", "texto": f"Inversion en infraestructura moderada (indice {barrio['infra']}/100)"})
    else:
        signals.append({"tipo": "negativo", "texto": f"Baja actividad de obra publica (indice {barrio['infra']}/100)"})

    if barrio["licencias"] >= 70:
        signals.append({"tipo": "positivo", "texto": f"Fuerte actividad promotora ({barrio['licencias']}/100)"})
    elif barrio["licencias"] >= 45:
        signals.append({"tipo": "neutro", "texto": f"Actividad promotora moderada ({barrio['licencias']}/100)"})
    else:
        signals.append({"tipo": "negativo", "texto": f"Escasa nueva construccion ({barrio['licencias']}/100)"})

    if barrio["metro"] >= 3:
        signals.append({"tipo": "positivo", "texto": f"{barrio['metro']} lineas de metro en radio 500m"})
    elif barrio["metro"] >= 2:
        signals.append({"tipo": "neutro", "texto": f"{barrio['metro']} lineas de metro proximas"})
    else:
        signals.append({"tipo": "negativo", "texto": "Acceso limitado a metro"})

    if barrio["tend_3a"] >= 24:
        signals.append({"tipo": "positivo", "texto": f"Tendencia sostenida: +{barrio['tend_3a']}% en 3 anos"})

    return signals


def _categoria(score: float) -> str:
    if score >= 72:
        return "Alta oportunidad"
    elif score >= 52:
        return "Oportunidad moderada"
    elif score >= 35:
        return "Media"
    else:
        return "Baja"


def _color(score: float) -> str:
    """Color HEX para el marcador del mapa."""
    if score >= 72:
        return "#64ffda"   # verde accent
    elif score >= 52:
        return "#a3e635"   # lima
    elif score >= 35:
        return "#fbbf24"   # amarillo
    else:
        return "#f87171"   # rojo


# ── Computar scores una sola vez al importar el modulo ────────────────────────

_barrios_index: Dict[str, Dict] = {b["id"]: b for b in BARRIOS}

# 1. Scores locales
_local: Dict[str, float] = {b["id"]: _local_score(b) for b in BARRIOS}

# 2. Round 1 de message-passing (alpha=0.65)
_r1 = _message_pass(_local, alpha=0.65)

# 3. Round 2 de message-passing (alpha=0.72)
_r2 = _message_pass(_r1, alpha=0.72)

# 4. Escalar a 0-100
_min_r2, _max_r2 = min(_r2.values()), max(_r2.values())

def _scale(v: float) -> float:
    return round(10 + 80 * (v - _min_r2) / (_max_r2 - _min_r2), 1)

SCORES: Dict[str, float] = {bid: _scale(v) for bid, v in _r2.items()}


# ── API publica del modulo ────────────────────────────────────────────────────

def get_all_barrios() -> List[Dict]:
    """Devuelve todos los barrios con su score y categoria (para el mapa)."""
    result = []
    for b in BARRIOS:
        score = SCORES[b["id"]]
        result.append({
            "id":        b["id"],
            "nombre":    b["nombre"],
            "lat":       b["lat"],
            "lng":       b["lng"],
            "score":     score,
            "categoria": _categoria(score),
            "color":     _color(score),
            "precio_m2": b["precio_m2"],
            "tend_1a":   b["tend_1a"],
        })
    return result


def get_barrio_detail(barrio_id: str) -> Dict:
    """Devuelve el analisis detallado de un barrio."""
    barrio = _barrios_index.get(barrio_id)
    if not barrio:
        return {}

    score = SCORES[barrio_id]
    neighbors = ADJACENCY.get(barrio_id, [])
    neighbor_data = [
        {
            "id":      n,
            "nombre":  _barrios_index[n]["nombre"],
            "score":   SCORES[n],
            "color":   _color(SCORES[n]),
        }
        for n in neighbors if n in _barrios_index
    ]

    # Score breakdown (contribucion de cada feature al score local)
    local = _local_score(barrio)
    breakdown = {
        "Tendencia 1 ano":   round(WEIGHTS["tend_1a"]    * _normalize(barrio["tend_1a"],   "tend_1a")    / local * 100, 1),
        "Infraestructura":   round(WEIGHTS["infra"]      * _normalize(barrio["infra"],     "infra")     / local * 100, 1),
        "Nuevas licencias":  round(WEIGHTS["licencias"]  * _normalize(barrio["licencias"],"licencias")  / local * 100, 1),
        "Acceso metro":      round(WEIGHTS["metro"]      * _normalize(barrio["metro"],     "metro")     / local * 100, 1),
        "Tendencia 3 anos":  round(WEIGHTS["tend_3a"]    * _normalize(barrio["tend_3a"],   "tend_3a")    / local * 100, 1),
    }

    return {
        "id":         barrio["id"],
        "nombre":     barrio["nombre"],
        "lat":        barrio["lat"],
        "lng":        barrio["lng"],
        "score":      score,
        "categoria":  _categoria(score),
        "color":      _color(score),
        # Features raw
        "precio_m2":  barrio["precio_m2"],
        "tend_1a":    barrio["tend_1a"],
        "tend_3a":    barrio["tend_3a"],
        "infra":      barrio["infra"],
        "metro":      barrio["metro"],
        "licencias":  barrio["licencias"],
        "renta":      barrio["renta"],
        # Analisis
        "signals":    _get_signals(barrio),
        "breakdown":  breakdown,
        "vecinos":    neighbor_data,
        "rounds_mp":  2,
    }


def get_stats() -> Dict:
    return {
        "modelo":             "Simplified Graph Neural Network (2-round message-passing)",
        "n_barrios":          len(BARRIOS),
        "n_aristas":          sum(len(v) for v in ADJACENCY.values()) // 2,
        "features":           list(WEIGHTS.keys()),
        "pesos":              WEIGHTS,
        "alpha_round1":       0.65,
        "alpha_round2":       0.72,
        "score_medio":        round(sum(SCORES.values()) / len(SCORES), 1),
        "score_max":          max(SCORES.values()),
        "score_min":          min(SCORES.values()),
        "top3":               sorted(
            [{"id": bid, "nombre": _barrios_index[bid]["nombre"], "score": s}
             for bid, s in SCORES.items()],
            key=lambda x: -x["score"]
        )[:3],
        "zona_alta_oportunidad": sum(1 for s in SCORES.values() if s >= 72),
    }

"""
Modelo de scoring de zonas de revalorizacion (multi-ciudad).
GNN simplificado: 2 rounds de message-passing sobre grafo de adyacencia geografica.
"""

import math
from typing import Dict, List

CIUDADES_META = {
    "madrid":    {"nombre": "Madrid",    "center": [40.416, -3.703], "zoom": 12},
    "barcelona": {"nombre": "Barcelona", "center": [41.385,  2.173], "zoom": 13},
    "valencia":  {"nombre": "Valencia",  "center": [39.470, -0.376], "zoom": 13},
    "sevilla":   {"nombre": "Sevilla",   "center": [37.384, -5.991], "zoom": 13},
    "bilbao":    {"nombre": "Bilbao",    "center": [43.263, -2.935], "zoom": 13},
    "zaragoza":  {"nombre": "Zaragoza",  "center": [41.651, -0.889], "zoom": 13},
    "malaga":    {"nombre": "Málaga",    "center": [36.717, -4.417], "zoom": 13},
}

_BARRIOS: Dict[str, List[Dict]] = {
    "madrid": [
        {"id": "malasana",      "nombre": "Malasaña",       "lat": 40.4268, "lng": -3.7035, "precio_m2": 5200, "tend_1a": 8.2,  "tend_3a": 22.1, "infra": 72, "metro": 3, "licencias": 68, "renta": 62},
        {"id": "lavapies",      "nombre": "Lavapiés",        "lat": 40.4082, "lng": -3.7035, "precio_m2": 3900, "tend_1a": 9.1,  "tend_3a": 25.3, "infra": 65, "metro": 2, "licencias": 71, "renta": 38},
        {"id": "chueca",        "nombre": "Chueca",          "lat": 40.4235, "lng": -3.6958, "precio_m2": 5400, "tend_1a": 6.8,  "tend_3a": 18.2, "infra": 55, "metro": 3, "licencias": 45, "renta": 70},
        {"id": "salamanca",     "nombre": "Salamanca",       "lat": 40.4262, "lng": -3.6820, "precio_m2": 7200, "tend_1a": 4.1,  "tend_3a": 12.3, "infra": 48, "metro": 3, "licencias": 30, "renta": 90},
        {"id": "chamberi",      "nombre": "Chamberí",        "lat": 40.4345, "lng": -3.6982, "precio_m2": 6100, "tend_1a": 5.2,  "tend_3a": 14.8, "infra": 52, "metro": 4, "licencias": 35, "renta": 80},
        {"id": "retiro",        "nombre": "Retiro",          "lat": 40.4082, "lng": -3.6820, "precio_m2": 6300, "tend_1a": 4.8,  "tend_3a": 13.1, "infra": 58, "metro": 2, "licencias": 28, "renta": 85},
        {"id": "arguelles",     "nombre": "Argüelles",       "lat": 40.4284, "lng": -3.7120, "precio_m2": 5600, "tend_1a": 5.9,  "tend_3a": 15.4, "infra": 61, "metro": 3, "licencias": 40, "renta": 72},
        {"id": "arganzuela",    "nombre": "Arganzuela",      "lat": 40.3990, "lng": -3.6940, "precio_m2": 4200, "tend_1a": 7.3,  "tend_3a": 19.8, "infra": 78, "metro": 2, "licencias": 62, "renta": 52},
        {"id": "carabanchel",   "nombre": "Carabanchel",     "lat": 40.3880, "lng": -3.7300, "precio_m2": 2800, "tend_1a": 10.2, "tend_3a": 28.7, "infra": 82, "metro": 3, "licencias": 80, "renta": 35},
        {"id": "vallecas",      "nombre": "Vallecas",        "lat": 40.3850, "lng": -3.6650, "precio_m2": 2400, "tend_1a": 11.5, "tend_3a": 31.2, "infra": 75, "metro": 2, "licencias": 78, "renta": 28},
        {"id": "tetuan",        "nombre": "Tetuán",          "lat": 40.4520, "lng": -3.6980, "precio_m2": 4100, "tend_1a": 9.8,  "tend_3a": 26.4, "infra": 88, "metro": 3, "licencias": 85, "renta": 45},
        {"id": "fuencarral",    "nombre": "Fuencarral",      "lat": 40.4800, "lng": -3.6930, "precio_m2": 3600, "tend_1a": 8.1,  "tend_3a": 21.3, "infra": 70, "metro": 2, "licencias": 72, "renta": 55},
        {"id": "hortaleza",     "nombre": "Hortaleza",       "lat": 40.4730, "lng": -3.6600, "precio_m2": 3800, "tend_1a": 7.2,  "tend_3a": 19.1, "infra": 65, "metro": 1, "licencias": 65, "renta": 58},
        {"id": "ciudad_lineal", "nombre": "Ciudad Lineal",   "lat": 40.4400, "lng": -3.6590, "precio_m2": 3500, "tend_1a": 6.8,  "tend_3a": 17.2, "infra": 60, "metro": 3, "licencias": 55, "renta": 50},
        {"id": "san_blas",      "nombre": "San Blas",        "lat": 40.4300, "lng": -3.6300, "precio_m2": 2900, "tend_1a": 5.9,  "tend_3a": 15.8, "infra": 52, "metro": 2, "licencias": 48, "renta": 40},
        {"id": "moratalaz",     "nombre": "Moratalaz",       "lat": 40.4050, "lng": -3.6510, "precio_m2": 2700, "tend_1a": 5.4,  "tend_3a": 14.2, "infra": 45, "metro": 1, "licencias": 40, "renta": 42},
        {"id": "vicalvaro",     "nombre": "Vicálvaro",       "lat": 40.4030, "lng": -3.6170, "precio_m2": 2300, "tend_1a": 7.1,  "tend_3a": 18.9, "infra": 68, "metro": 1, "licencias": 70, "renta": 32},
        {"id": "usera",         "nombre": "Usera",           "lat": 40.3900, "lng": -3.7050, "precio_m2": 2600, "tend_1a": 6.2,  "tend_3a": 16.5, "infra": 55, "metro": 2, "licencias": 52, "renta": 33},
        {"id": "latina",        "nombre": "Latina",          "lat": 40.4040, "lng": -3.7260, "precio_m2": 2900, "tend_1a": 5.8,  "tend_3a": 15.3, "infra": 50, "metro": 3, "licencias": 45, "renta": 38},
        {"id": "villaverde",    "nombre": "Villaverde",      "lat": 40.3600, "lng": -3.7000, "precio_m2": 1900, "tend_1a": 4.5,  "tend_3a": 12.1, "infra": 38, "metro": 1, "licencias": 35, "renta": 22},
        {"id": "barajas",       "nombre": "Barajas",         "lat": 40.4740, "lng": -3.5930, "precio_m2": 2800, "tend_1a": 4.2,  "tend_3a": 11.3, "infra": 42, "metro": 1, "licencias": 32, "renta": 45},
        {"id": "palacio",       "nombre": "Palacio",         "lat": 40.4143, "lng": -3.7110, "precio_m2": 5800, "tend_1a": 5.5,  "tend_3a": 14.8, "infra": 62, "metro": 2, "licencias": 38, "renta": 68},
        {"id": "moncloa",       "nombre": "Moncloa",         "lat": 40.4350, "lng": -3.7200, "precio_m2": 5100, "tend_1a": 5.1,  "tend_3a": 13.5, "infra": 55, "metro": 2, "licencias": 36, "renta": 72},
    ],
    "barcelona": [
        {"id": "eixample_esq",  "nombre": "Eixample Esquerra", "lat": 41.3810, "lng": 2.1580, "precio_m2": 5800, "tend_1a": 6.2,  "tend_3a": 17.4, "infra": 62, "metro": 4, "licencias": 42, "renta": 75},
        {"id": "eixample_dret", "nombre": "Eixample Dreta",    "lat": 41.3920, "lng": 2.1680, "precio_m2": 6400, "tend_1a": 5.1,  "tend_3a": 13.8, "infra": 55, "metro": 5, "licencias": 33, "renta": 88},
        {"id": "gracia",        "nombre": "Gràcia",             "lat": 41.4020, "lng": 2.1580, "precio_m2": 5200, "tend_1a": 7.4,  "tend_3a": 20.1, "infra": 60, "metro": 3, "licencias": 55, "renta": 70},
        {"id": "poblenou",      "nombre": "Poblenou (22@)",     "lat": 41.3990, "lng": 2.1970, "precio_m2": 5000, "tend_1a": 10.3, "tend_3a": 29.5, "infra": 90, "metro": 2, "licencias": 88, "renta": 65},
        {"id": "gothic",        "nombre": "Barri Gòtic",        "lat": 41.3826, "lng": 2.1769, "precio_m2": 5600, "tend_1a": 4.8,  "tend_3a": 12.9, "infra": 52, "metro": 3, "licencias": 28, "renta": 60},
        {"id": "born",          "nombre": "El Born",            "lat": 41.3850, "lng": 2.1830, "precio_m2": 5900, "tend_1a": 6.8,  "tend_3a": 18.7, "infra": 64, "metro": 2, "licencias": 48, "renta": 72},
        {"id": "barceloneta",   "nombre": "Barceloneta",        "lat": 41.3797, "lng": 2.1876, "precio_m2": 5100, "tend_1a": 5.6,  "tend_3a": 14.3, "infra": 58, "metro": 2, "licencias": 38, "renta": 52},
        {"id": "sarria",        "nombre": "Sarrià",             "lat": 41.4020, "lng": 2.1140, "precio_m2": 7200, "tend_1a": 3.9,  "tend_3a": 10.8, "infra": 45, "metro": 2, "licencias": 26, "renta": 95},
        {"id": "les_corts",     "nombre": "Les Corts",          "lat": 41.3840, "lng": 2.1320, "precio_m2": 5500, "tend_1a": 5.4,  "tend_3a": 14.1, "infra": 55, "metro": 3, "licencias": 35, "renta": 80},
        {"id": "sants",         "nombre": "Sants",              "lat": 41.3760, "lng": 2.1380, "precio_m2": 4200, "tend_1a": 7.8,  "tend_3a": 21.6, "infra": 72, "metro": 4, "licencias": 65, "renta": 55},
        {"id": "sant_andreu",   "nombre": "Sant Andreu",        "lat": 41.4360, "lng": 2.1870, "precio_m2": 3600, "tend_1a": 9.1,  "tend_3a": 25.2, "infra": 78, "metro": 2, "licencias": 75, "renta": 42},
        {"id": "nou_barris",    "nombre": "Nou Barris",         "lat": 41.4400, "lng": 2.1760, "precio_m2": 2800, "tend_1a": 10.5, "tend_3a": 28.8, "infra": 82, "metro": 2, "licencias": 80, "renta": 30},
        {"id": "horta",         "nombre": "Horta",              "lat": 41.4320, "lng": 2.1600, "precio_m2": 3200, "tend_1a": 8.2,  "tend_3a": 22.0, "infra": 68, "metro": 1, "licencias": 62, "renta": 38},
        {"id": "sant_marti",    "nombre": "Sant Martí",         "lat": 41.4050, "lng": 2.1990, "precio_m2": 4400, "tend_1a": 8.9,  "tend_3a": 24.3, "infra": 75, "metro": 3, "licencias": 72, "renta": 58},
        {"id": "raval",         "nombre": "El Raval",           "lat": 41.3795, "lng": 2.1660, "precio_m2": 4000, "tend_1a": 7.1,  "tend_3a": 19.2, "infra": 68, "metro": 3, "licencias": 58, "renta": 40},
    ],
    "valencia": [
        {"id": "ruzafa",       "nombre": "Ruzafa",           "lat": 39.4620, "lng": -0.3770, "precio_m2": 3200, "tend_1a": 11.2, "tend_3a": 30.5, "infra": 82, "metro": 2, "licencias": 85, "renta": 55},
        {"id": "cabanyal",     "nombre": "Cabanyal",         "lat": 39.4710, "lng": -0.3260, "precio_m2": 2100, "tend_1a": 13.5, "tend_3a": 38.2, "infra": 88, "metro": 1, "licencias": 90, "renta": 32},
        {"id": "eixample_vlc", "nombre": "L'Eixample",       "lat": 39.4680, "lng": -0.3820, "precio_m2": 3800, "tend_1a": 7.8,  "tend_3a": 21.0, "infra": 60, "metro": 3, "licencias": 52, "renta": 72},
        {"id": "benimaclet",   "nombre": "Benimaclet",       "lat": 39.4840, "lng": -0.3680, "precio_m2": 2400, "tend_1a": 9.6,  "tend_3a": 26.3, "infra": 70, "metro": 2, "licencias": 72, "renta": 45},
        {"id": "la_seu",       "nombre": "La Seu",           "lat": 39.4745, "lng": -0.3755, "precio_m2": 3600, "tend_1a": 5.2,  "tend_3a": 14.1, "infra": 55, "metro": 2, "licencias": 35, "renta": 65},
        {"id": "el_carmen",    "nombre": "El Carme",         "lat": 39.4770, "lng": -0.3800, "precio_m2": 3100, "tend_1a": 7.3,  "tend_3a": 19.8, "infra": 65, "metro": 2, "licencias": 55, "renta": 58},
        {"id": "quatre_carr",  "nombre": "Quatre Carreres",  "lat": 39.4530, "lng": -0.3720, "precio_m2": 2000, "tend_1a": 8.1,  "tend_3a": 22.4, "infra": 68, "metro": 1, "licencias": 65, "renta": 35},
        {"id": "campanar",     "nombre": "Campanar",         "lat": 39.4850, "lng": -0.4000, "precio_m2": 2600, "tend_1a": 6.4,  "tend_3a": 17.5, "infra": 58, "metro": 2, "licencias": 48, "renta": 50},
        {"id": "patraix",      "nombre": "Patraix",          "lat": 39.4590, "lng": -0.3960, "precio_m2": 1800, "tend_1a": 7.0,  "tend_3a": 19.0, "infra": 52, "metro": 1, "licencias": 55, "renta": 38},
        {"id": "olivereta",    "nombre": "L'Olivereta",      "lat": 39.4650, "lng": -0.4020, "precio_m2": 1600, "tend_1a": 5.8,  "tend_3a": 15.6, "infra": 45, "metro": 1, "licencias": 42, "renta": 30},
        {"id": "torrefiel",    "nombre": "Torrefiel",        "lat": 39.4930, "lng": -0.3690, "precio_m2": 1500, "tend_1a": 6.2,  "tend_3a": 16.8, "infra": 48, "metro": 1, "licencias": 45, "renta": 28},
        {"id": "malvarrosa",   "nombre": "La Malva-rosa",    "lat": 39.4760, "lng": -0.3200, "precio_m2": 2300, "tend_1a": 9.8,  "tend_3a": 26.9, "infra": 72, "metro": 1, "licencias": 70, "renta": 40},
    ],
    "sevilla": [
        {"id": "triana",       "nombre": "Triana",           "lat": 37.3870, "lng": -6.0020, "precio_m2": 2800, "tend_1a": 9.2,  "tend_3a": 25.1, "infra": 75, "metro": 1, "licencias": 72, "renta": 52},
        {"id": "santa_cruz",   "nombre": "Santa Cruz",       "lat": 37.3862, "lng": -5.9904, "precio_m2": 3800, "tend_1a": 5.8,  "tend_3a": 15.3, "infra": 55, "metro": 1, "licencias": 32, "renta": 72},
        {"id": "nervion",      "nombre": "Nervión",          "lat": 37.3849, "lng": -5.9700, "precio_m2": 3200, "tend_1a": 6.4,  "tend_3a": 17.2, "infra": 60, "metro": 1, "licencias": 45, "renta": 75},
        {"id": "los_remedios", "nombre": "Los Remedios",     "lat": 37.3750, "lng": -5.9990, "precio_m2": 3500, "tend_1a": 5.1,  "tend_3a": 13.8, "infra": 52, "metro": 1, "licencias": 35, "renta": 80},
        {"id": "macarena",     "nombre": "Macarena",         "lat": 37.4030, "lng": -5.9893, "precio_m2": 1900, "tend_1a": 10.5, "tend_3a": 28.7, "infra": 80, "metro": 1, "licencias": 82, "renta": 32},
        {"id": "heliop",       "nombre": "Heliópolis",       "lat": 37.3630, "lng": -6.0030, "precio_m2": 3100, "tend_1a": 6.8,  "tend_3a": 18.4, "infra": 62, "metro": 1, "licencias": 50, "renta": 70},
        {"id": "san_pablo",    "nombre": "San Pablo",        "lat": 37.4120, "lng": -5.9800, "precio_m2": 1600, "tend_1a": 8.9,  "tend_3a": 24.1, "infra": 72, "metro": 1, "licencias": 78, "renta": 28},
        {"id": "palmete",      "nombre": "Palmete",          "lat": 37.3680, "lng": -5.9620, "precio_m2": 1400, "tend_1a": 7.4,  "tend_3a": 20.0, "infra": 65, "metro": 1, "licencias": 68, "renta": 25},
        {"id": "cerro_amate",  "nombre": "Cerro-Amate",      "lat": 37.3700, "lng": -5.9760, "precio_m2": 1300, "tend_1a": 6.8,  "tend_3a": 18.5, "infra": 60, "metro": 1, "licencias": 62, "renta": 22},
        {"id": "el_porvenir",  "nombre": "El Porvenir",      "lat": 37.3780, "lng": -5.9820, "precio_m2": 3400, "tend_1a": 5.5,  "tend_3a": 14.8, "infra": 55, "metro": 1, "licencias": 38, "renta": 78},
    ],
    "bilbao": [
        {"id": "casco_viejo",  "nombre": "Casco Viejo",      "lat": 43.2590, "lng": -2.9220, "precio_m2": 3400, "tend_1a": 7.2,  "tend_3a": 19.5, "infra": 65, "metro": 2, "licencias": 55, "renta": 60},
        {"id": "deusto",       "nombre": "Deusto",           "lat": 43.2720, "lng": -2.9420, "precio_m2": 2600, "tend_1a": 9.1,  "tend_3a": 24.8, "infra": 75, "metro": 2, "licencias": 72, "renta": 55},
        {"id": "indautxu",     "nombre": "Indautxu",         "lat": 43.2638, "lng": -2.9365, "precio_m2": 4200, "tend_1a": 5.8,  "tend_3a": 15.4, "infra": 52, "metro": 3, "licencias": 38, "renta": 85},
        {"id": "abando",       "nombre": "Abando",           "lat": 43.2620, "lng": -2.9295, "precio_m2": 3800, "tend_1a": 6.4,  "tend_3a": 17.2, "infra": 58, "metro": 3, "licencias": 42, "renta": 80},
        {"id": "rekalde",      "nombre": "Rekalde",          "lat": 43.2520, "lng": -2.9350, "precio_m2": 2200, "tend_1a": 10.8, "tend_3a": 29.5, "infra": 82, "metro": 2, "licencias": 85, "renta": 38},
        {"id": "begona",       "nombre": "Begoña",           "lat": 43.2680, "lng": -2.9150, "precio_m2": 2900, "tend_1a": 7.8,  "tend_3a": 21.2, "infra": 68, "metro": 1, "licencias": 62, "renta": 58},
        {"id": "basurto",      "nombre": "Basurto",          "lat": 43.2578, "lng": -2.9480, "precio_m2": 3000, "tend_1a": 6.9,  "tend_3a": 18.7, "infra": 62, "metro": 2, "licencias": 52, "renta": 65},
        {"id": "otxarkoaga",   "nombre": "Otxarkoaga",       "lat": 43.2780, "lng": -2.9280, "precio_m2": 1800, "tend_1a": 8.5,  "tend_3a": 23.1, "infra": 70, "metro": 1, "licencias": 75, "renta": 30},
    ],
    "zaragoza": [
        {"id": "centro_zgz",   "nombre": "Centro",           "lat": 41.6561, "lng": -0.8773, "precio_m2": 2400, "tend_1a": 6.8,  "tend_3a": 18.2, "infra": 60, "metro": 1, "licencias": 48, "renta": 65},
        {"id": "delicias",     "nombre": "Delicias",         "lat": 41.6432, "lng": -0.9010, "precio_m2": 1600, "tend_1a": 9.4,  "tend_3a": 25.6, "infra": 78, "metro": 1, "licencias": 80, "renta": 38},
        {"id": "las_fuentes",  "nombre": "Las Fuentes",      "lat": 41.6490, "lng": -0.8620, "precio_m2": 1500, "tend_1a": 8.2,  "tend_3a": 22.1, "infra": 72, "metro": 1, "licencias": 72, "renta": 35},
        {"id": "oliver",       "nombre": "Oliver",           "lat": 41.6578, "lng": -0.9180, "precio_m2": 1200, "tend_1a": 7.6,  "tend_3a": 20.5, "infra": 68, "metro": 1, "licencias": 70, "renta": 28},
        {"id": "torrero",      "nombre": "Torrero",          "lat": 41.6353, "lng": -0.8810, "precio_m2": 1400, "tend_1a": 8.8,  "tend_3a": 23.9, "infra": 74, "metro": 1, "licencias": 75, "renta": 32},
        {"id": "miralbueno",   "nombre": "Miralbueno",       "lat": 41.6620, "lng": -0.9350, "precio_m2": 1800, "tend_1a": 7.1,  "tend_3a": 19.2, "infra": 65, "metro": 1, "licencias": 62, "renta": 48},
        {"id": "el_rabal_zgz", "nombre": "El Rabal",         "lat": 41.6680, "lng": -0.8780, "precio_m2": 1300, "tend_1a": 10.2, "tend_3a": 27.8, "infra": 80, "metro": 1, "licencias": 82, "renta": 30},
        {"id": "casablanca",   "nombre": "Casablanca",       "lat": 41.6280, "lng": -0.9050, "precio_m2": 1700, "tend_1a": 5.4,  "tend_3a": 14.6, "infra": 50, "metro": 1, "licencias": 42, "renta": 42},
    ],
    "malaga": [
        {"id": "centro_mlg",   "nombre": "Centro Histórico", "lat": 36.7213, "lng": -4.4215, "precio_m2": 3200, "tend_1a": 8.5,  "tend_3a": 23.1, "infra": 68, "metro": 1, "licencias": 62, "renta": 65},
        {"id": "soho_mlg",     "nombre": "Soho",             "lat": 36.7162, "lng": -4.4270, "precio_m2": 3000, "tend_1a": 12.4, "tend_3a": 33.8, "infra": 85, "metro": 1, "licencias": 88, "renta": 58},
        {"id": "lagunillas",   "nombre": "Lagunillas",       "lat": 36.7260, "lng": -4.4190, "precio_m2": 2000, "tend_1a": 9.6,  "tend_3a": 26.2, "infra": 72, "metro": 1, "licencias": 75, "renta": 42},
        {"id": "cruz_humil",   "nombre": "Cruz de Humilladero","lat": 36.7120, "lng": -4.4400, "precio_m2": 1700, "tend_1a": 8.1, "tend_3a": 22.0, "infra": 68, "metro": 1, "licencias": 70, "renta": 35},
        {"id": "campanillas",  "nombre": "Campanillas",      "lat": 36.7350, "lng": -4.5200, "precio_m2": 1200, "tend_1a": 6.2,  "tend_3a": 16.8, "infra": 52, "metro": 1, "licencias": 55, "renta": 30},
        {"id": "churriana",    "nombre": "Churriana",        "lat": 36.6910, "lng": -4.5100, "precio_m2": 1500, "tend_1a": 7.4,  "tend_3a": 20.1, "infra": 58, "metro": 1, "licencias": 60, "renta": 38},
        {"id": "el_palo",      "nombre": "El Palo",          "lat": 36.7130, "lng": -4.3740, "precio_m2": 2600, "tend_1a": 9.2,  "tend_3a": 25.0, "infra": 75, "metro": 1, "licencias": 72, "renta": 52},
        {"id": "pedregalejo",  "nombre": "Pedregalejo",      "lat": 36.7175, "lng": -4.3860, "precio_m2": 3000, "tend_1a": 7.8,  "tend_3a": 21.2, "infra": 65, "metro": 1, "licencias": 55, "renta": 60},
    ],
}

_ADJ: Dict[str, Dict[str, List[str]]] = {
    "madrid": {
        "malasana": ["chueca","chamberi","palacio","arguelles"],
        "lavapies": ["palacio","arganzuela","retiro","usera"],
        "chueca": ["malasana","chamberi","salamanca"],
        "salamanca": ["chueca","chamberi","retiro","ciudad_lineal"],
        "chamberi": ["malasana","chueca","tetuan","arguelles","salamanca"],
        "retiro": ["salamanca","lavapies","arganzuela","moratalaz"],
        "arguelles": ["malasana","chamberi","moncloa","palacio"],
        "arganzuela": ["lavapies","retiro","usera","vallecas","carabanchel"],
        "carabanchel": ["arganzuela","latina","usera","villaverde"],
        "vallecas": ["arganzuela","moratalaz","vicalvaro","usera"],
        "tetuan": ["chamberi","fuencarral","ciudad_lineal"],
        "fuencarral": ["tetuan","hortaleza"],
        "hortaleza": ["fuencarral","ciudad_lineal","san_blas","barajas"],
        "ciudad_lineal": ["tetuan","hortaleza","san_blas","salamanca"],
        "san_blas": ["ciudad_lineal","hortaleza","moratalaz","vicalvaro"],
        "moratalaz": ["retiro","san_blas","vallecas","vicalvaro"],
        "vicalvaro": ["moratalaz","vallecas","san_blas"],
        "usera": ["lavapies","arganzuela","carabanchel","villaverde"],
        "latina": ["palacio","carabanchel","arguelles"],
        "villaverde": ["carabanchel","usera"],
        "barajas": ["hortaleza"],
        "palacio": ["malasana","lavapies","latina","arguelles"],
        "moncloa": ["arguelles","chamberi"],
    },
    "barcelona": {
        "eixample_esq": ["eixample_dret","gracia","sants","gothic","raval"],
        "eixample_dret": ["eixample_esq","gracia","born","gothic"],
        "gracia": ["eixample_esq","eixample_dret","horta","sant_marti"],
        "poblenou": ["sant_marti","barceloneta","born"],
        "gothic": ["eixample_esq","eixample_dret","born","barceloneta","raval"],
        "born": ["gothic","barceloneta","poblenou","eixample_dret"],
        "barceloneta": ["gothic","born","poblenou"],
        "sarria": ["les_corts","gracia"],
        "les_corts": ["sarria","eixample_esq","sants"],
        "sants": ["les_corts","eixample_esq","raval"],
        "sant_andreu": ["nou_barris","horta","sant_marti"],
        "nou_barris": ["sant_andreu","horta"],
        "horta": ["gracia","sant_andreu","nou_barris"],
        "sant_marti": ["gracia","poblenou","sant_andreu"],
        "raval": ["gothic","eixample_esq","sants"],
    },
    "valencia": {
        "ruzafa": ["eixample_vlc","quatre_carr","la_seu"],
        "cabanyal": ["malvarrosa","benimaclet"],
        "eixample_vlc": ["ruzafa","la_seu","el_carmen","campanar"],
        "benimaclet": ["cabanyal","torrefiel","eixample_vlc"],
        "la_seu": ["eixample_vlc","el_carmen","ruzafa"],
        "el_carmen": ["la_seu","eixample_vlc","campanar"],
        "quatre_carr": ["ruzafa","patraix","olivereta"],
        "campanar": ["eixample_vlc","el_carmen","patraix"],
        "patraix": ["quatre_carr","campanar","olivereta"],
        "olivereta": ["quatre_carr","patraix"],
        "torrefiel": ["benimaclet"],
        "malvarrosa": ["cabanyal"],
    },
    "sevilla": {
        "triana":       ["santa_cruz","los_remedios","heliop"],
        "santa_cruz":   ["triana","nervion","el_porvenir"],
        "nervion":      ["santa_cruz","el_porvenir","macarena"],
        "los_remedios": ["triana","heliop","el_porvenir"],
        "macarena":     ["nervion","san_pablo"],
        "heliop":       ["triana","los_remedios","palmete","cerro_amate"],
        "san_pablo":    ["macarena","palmete"],
        "palmete":      ["san_pablo","cerro_amate","heliop"],
        "cerro_amate":  ["palmete","heliop","el_porvenir"],
        "el_porvenir":  ["santa_cruz","nervion","los_remedios","cerro_amate"],
    },
    "bilbao": {
        "casco_viejo":  ["abando","begona","deusto"],
        "deusto":       ["casco_viejo","abando","basurto","otxarkoaga"],
        "indautxu":     ["abando","rekalde","basurto"],
        "abando":       ["casco_viejo","deusto","indautxu","rekalde"],
        "rekalde":      ["abando","indautxu","basurto"],
        "begona":       ["casco_viejo","otxarkoaga"],
        "basurto":      ["deusto","indautxu","rekalde"],
        "otxarkoaga":   ["deusto","begona"],
    },
    "zaragoza": {
        "centro_zgz":   ["delicias","las_fuentes","el_rabal_zgz"],
        "delicias":     ["centro_zgz","oliver","miralbueno","torrero"],
        "las_fuentes":  ["centro_zgz","torrero"],
        "oliver":       ["delicias","miralbueno"],
        "torrero":      ["delicias","las_fuentes","casablanca"],
        "miralbueno":   ["delicias","oliver"],
        "el_rabal_zgz": ["centro_zgz"],
        "casablanca":   ["torrero"],
    },
    "malaga": {
        "centro_mlg":   ["soho_mlg","lagunillas","pedregalejo"],
        "soho_mlg":     ["centro_mlg","cruz_humil"],
        "lagunillas":   ["centro_mlg","el_palo"],
        "cruz_humil":   ["soho_mlg","campanillas","churriana"],
        "campanillas":  ["cruz_humil","churriana"],
        "churriana":    ["cruz_humil","campanillas"],
        "el_palo":      ["lagunillas","pedregalejo"],
        "pedregalejo":  ["centro_mlg","el_palo"],
    },
}

WEIGHTS = {"tend_1a": 0.30, "infra": 0.25, "licencias": 0.20, "metro": 0.15, "tend_3a": 0.10}

_cache: Dict[str, Dict] = {}


def _build_ranges(barrios):
    return {f: (min(b[f] for b in barrios), max(b[f] for b in barrios)) for f in WEIGHTS}


def _norm(v, f, ranges):
    lo, hi = ranges[f]
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi != lo else 0.5


def _local(b, ranges):
    return sum(WEIGHTS[f] * _norm(b[f], f, ranges) for f in WEIGHTS)


def _mp(scores, adj, alpha):
    return {bid: alpha * s + (1 - alpha) * (sum(scores.get(n, s) for n in adj.get(bid, [])) / len(adj.get(bid, [bid])))
            for bid, s in scores.items()}


def _categoria(s):
    return "Alta oportunidad" if s >= 72 else "Oportunidad moderada" if s >= 52 else "Media" if s >= 35 else "Baja"


def _color(s):
    return "#64ffda" if s >= 72 else "#a3e635" if s >= 52 else "#fbbf24" if s >= 35 else "#f87171"


def _get_signals(barrio):
    signals = []
    t = barrio["tend_1a"]
    signals.append({"tipo": "positivo" if t >= 9 else "neutro" if t >= 6 else "negativo",
                     "texto": f"Revalorización anual: +{t}%"})
    i = barrio["infra"]
    signals.append({"tipo": "positivo" if i >= 75 else "neutro" if i >= 55 else "negativo",
                     "texto": f"Inversión infraestructura: {i}/100"})
    l = barrio["licencias"]
    signals.append({"tipo": "positivo" if l >= 70 else "neutro" if l >= 45 else "negativo",
                     "texto": f"Actividad promotora: {l}/100"})
    m = barrio["metro"]
    signals.append({"tipo": "positivo" if m >= 3 else "neutro" if m >= 2 else "negativo",
                     "texto": f"{m} líneas de transporte en radio 500m"})
    if barrio["tend_3a"] >= 24:
        signals.append({"tipo": "positivo", "texto": f"Tendencia sostenida: +{barrio['tend_3a']}% en 3 años"})
    return signals


def _compute(ciudad):
    if ciudad in _cache:
        return _cache[ciudad]
    barrios = _BARRIOS.get(ciudad, [])
    if not barrios:
        return {}
    adj = _ADJ.get(ciudad, {})
    ranges = _build_ranges(barrios)
    loc = {b["id"]: _local(b, ranges) for b in barrios}
    r1 = _mp(loc, adj, 0.65)
    r2 = _mp(r1, adj, 0.72)
    mn, mx = min(r2.values()), max(r2.values())
    scores = {bid: round(10 + 80 * (v - mn) / (mx - mn) if mx != mn else 50, 1) for bid, v in r2.items()}
    _cache[ciudad] = {"scores": scores, "index": {b["id"]: b for b in barrios}, "adj": adj, "ranges": ranges}
    return _cache[ciudad]


# ── API pública ────────────────────────────────────────────────────────────────

def get_ciudades():
    return [{"id": cid, **meta} for cid, meta in CIUDADES_META.items()]


def get_all_barrios(ciudad="madrid"):
    d = _compute(ciudad)
    if not d:
        return []
    return [{"id": b["id"], "nombre": b["nombre"], "lat": b["lat"], "lng": b["lng"],
              "score": d["scores"][b["id"]], "categoria": _categoria(d["scores"][b["id"]]),
              "color": _color(d["scores"][b["id"]]), "precio_m2": b["precio_m2"], "tend_1a": b["tend_1a"]}
             for b in _BARRIOS[ciudad]]


def get_barrio_detail(barrio_id, ciudad="madrid"):
    d = _compute(ciudad)
    if not d:
        return {}
    b = d["index"].get(barrio_id)
    if not b:
        return {}
    s = d["scores"][barrio_id]
    loc = _local(b, d["ranges"])
    breakdown = {f"Tend. 1 año": round(WEIGHTS["tend_1a"] * _norm(b["tend_1a"], "tend_1a", d["ranges"]) / loc * 100, 1),
                  "Infraestructura": round(WEIGHTS["infra"] * _norm(b["infra"], "infra", d["ranges"]) / loc * 100, 1),
                  "Nuevas licencias": round(WEIGHTS["licencias"] * _norm(b["licencias"], "licencias", d["ranges"]) / loc * 100, 1),
                  "Acceso transporte": round(WEIGHTS["metro"] * _norm(b["metro"], "metro", d["ranges"]) / loc * 100, 1),
                  "Tend. 3 años": round(WEIGHTS["tend_3a"] * _norm(b["tend_3a"], "tend_3a", d["ranges"]) / loc * 100, 1)}
    vecinos = [{"id": n, "nombre": d["index"][n]["nombre"], "score": d["scores"][n], "color": _color(d["scores"][n])}
               for n in d["adj"].get(barrio_id, []) if n in d["index"]]
    return {**b, "score": s, "categoria": _categoria(s), "color": _color(s),
            "signals": _get_signals(b), "breakdown": breakdown, "vecinos": vecinos, "rounds_mp": 2}


def get_stats(ciudad="madrid"):
    d = _compute(ciudad)
    if not d:
        return {}
    scores = d["scores"]
    meta = CIUDADES_META.get(ciudad, {})
    return {"ciudad": ciudad, "nombre_ciudad": meta.get("nombre", ciudad),
            "modelo": "Simplified GNN (2-round message-passing)",
            "n_barrios": len(scores), "n_aristas": sum(len(v) for v in d["adj"].values()) // 2,
            "features": list(WEIGHTS.keys()), "pesos": WEIGHTS,
            "alpha_round1": 0.65, "alpha_round2": 0.72,
            "score_medio": round(sum(scores.values()) / len(scores), 1),
            "score_max": max(scores.values()), "score_min": min(scores.values()),
            "top3": sorted([{"id": bid, "nombre": d["index"][bid]["nombre"], "score": s}
                            for bid, s in scores.items()], key=lambda x: -x["score"])[:3],
            "zona_alta_oportunidad": sum(1 for s in scores.values() if s >= 72)}

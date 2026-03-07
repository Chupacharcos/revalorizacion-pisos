"""
Router ML — Deteccion de zonas de revalorizacion.
Se carga dinamicamente desde /var/www/chatbot/src/api.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/ml", tags=["ML"])


@router.get("/revalorizacion/mapa")
def revalorizacion_mapa():
    """
    Devuelve todos los barrios con score y coordenadas para renderizar el mapa.
    Respuesta ligera: solo los campos necesarios para el frontend.
    """
    from data import get_all_barrios
    return {"barrios": get_all_barrios()}


@router.get("/revalorizacion/barrio/{barrio_id}")
def revalorizacion_barrio(barrio_id: str):
    """
    Analisis detallado de un barrio: features, score breakdown, senales, vecinos.
    """
    from data import get_barrio_detail
    detail = get_barrio_detail(barrio_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Barrio '{barrio_id}' no encontrado")
    return detail


@router.get("/revalorizacion/stats")
def revalorizacion_stats():
    """Metadatos del modelo GNN para mostrar en la demo."""
    from data import get_stats
    return get_stats()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import APIRouter, HTTPException, Query
router = APIRouter(prefix="/ml", tags=["ML"])

@router.get("/revalorizacion/ciudades")
def ciudades():
    from data import get_ciudades
    return {"ciudades": get_ciudades()}

@router.get("/revalorizacion/mapa")
def mapa(ciudad: str = Query("madrid")):
    from data import get_all_barrios, CIUDADES_META
    if ciudad not in CIUDADES_META:
        raise HTTPException(status_code=400, detail=f"Ciudad '{ciudad}' no disponible")
    meta = CIUDADES_META[ciudad]
    return {"barrios": get_all_barrios(ciudad), "ciudad": ciudad,
            "nombre_ciudad": meta["nombre"], "center": meta["center"], "zoom": meta["zoom"]}

@router.get("/revalorizacion/barrio/{barrio_id}")
def barrio(barrio_id: str, ciudad: str = Query("madrid")):
    from data import get_barrio_detail
    detail = get_barrio_detail(barrio_id, ciudad)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Barrio '{barrio_id}' no encontrado en {ciudad}")
    return detail

@router.get("/revalorizacion/stats")
def stats(ciudad: str = Query("madrid")):
    from data import get_stats
    return get_stats(ciudad)

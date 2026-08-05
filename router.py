import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
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

@router.get("/revalorizacion/geojson")
def geojson(ciudad: str = Query("madrid")):
    """Los barrios y su score de revalorización en GeoJSON (RFC 7946).

    Pensado para integrarlo en un GIS o en el mapa de un portal inmobiliario:
    QGIS, ArcGIS, Leaflet o Mapbox lo cargan por URL y pueden colorear
    directamente por la propiedad `score` sin conversión previa.
    """
    from data import get_all_barrios, CIUDADES_META
    if ciudad not in CIUDADES_META:
        raise HTTPException(status_code=400, detail=f"Ciudad '{ciudad}' no disponible")

    features = [
        {
            "type": "Feature",
            "id": b["id"],
            "geometry": {"type": "Point", "coordinates": [b["lng"], b["lat"]]},  # GeoJSON: lon, lat
            "properties": {k: v for k, v in b.items() if k not in ("lat", "lng")},
        }
        for b in get_all_barrios(ciudad)
    ]
    return JSONResponse(
        content={
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"ciudad": ciudad, "nombre_ciudad": CIUDADES_META[ciudad]["nombre"],
                         "fuente": "deteccion-zonas-revalorizacion"},
        },
        media_type="application/geo+json",
    )


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


@router.get("/revalorizacion/data-sources")
def data_sources():
    """Fuentes de datos activas (INE, Catastro, Idealista). Idealista se activa
    cuando IDEALISTA_API_KEY se configure en el .env."""
    try:
        from data_sources import get_active_sources, SUPPORTED_CITIES
        active = get_active_sources()
        return {
            "active_sources": active,
            "supported_cities": SUPPORTED_CITIES,
            "idealista_enabled": "Idealista" in active,
        }
    except Exception as e:
        return {"active_sources": ["sintético"], "error": str(e)}

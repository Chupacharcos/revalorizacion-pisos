import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
router = APIRouter(prefix="/visualizacion", tags=["Visualización"])
@router.get("/revalorizacion")
def revalorizacion():
    from data import get_all_barrios
    barrios = get_all_barrios()
    return JSONResponse(content=barrios, media_type="application/json")
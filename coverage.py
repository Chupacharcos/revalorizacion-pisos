from pathlib import Path
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(prefix="/ml", tags=["coverage"])

ARTIFACTS = Path(__file__).parent / "artifacts"

def _count_barrios() -> int:
    if not ARTIFACTS.is_dir():
        return 0
    try:
        jsons = [f for f in ARTIFACTS.iterdir() if f.suffix == ".json"]
        return len(jsons)
    except Exception:
        return 0

@router.get("/coverage")
def coverage():
    info = {
        "service": "ml-deteccion-zonas-revalorizacion",
        "ciudades": ["madrid", "barcelona", "valencia", "sevilla"],
        "barrios_analizados": _count_barrios(),
        "metodologia": "GNN simplificado (2-round message passing) sobre grafo barrio-vecindad",
        "ventana_temporal_anyos": 5,
        "actualizacion": "semanal",
    }
    if ARTIFACTS.is_dir():
        last = max((f for f in ARTIFACTS.iterdir() if f.is_file()), key=lambda p: p.stat().st_mtime, default=None)
        if last:
            info["last_artifact_at"] = datetime.fromtimestamp(last.stat().st_mtime).isoformat()
    return info
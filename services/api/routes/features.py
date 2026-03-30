from fastapi import APIRouter, HTTPException
from feast_client import get_features

router = APIRouter()

@router.get("/features/{user_id}")
def get_features_api(user_id: int):
    try:
        features = get_features(user_id)
    except Exception:
        raise HTTPException(status_code=404, detail="No features found")

    return {
        "user_id": user_id,
        "features": features
    }
from fastapi import APIRouter, HTTPException
from feast_client import get_features
from onnx_model import predict as onnx_predict

router = APIRouter()

@router.get("/predict/{user_id}")
def predict(user_id: int):
    try:
        features = get_features(user_id)
    except Exception:
        raise HTTPException(status_code=404, detail="No features found")

    result = onnx_predict(features)

    return {
        "user_id": user_id,
        "features": features,
        "prediction": result
    }
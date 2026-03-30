import onnxruntime as ort
import numpy as np

MODEL_PATH = "/app/model.onnx"

session = None

def load_model():
    global session
    session = ort.InferenceSession(MODEL_PATH)


def predict(features: dict):
    txn_count = float(features.get("txn_count_1min", 0))
    txn_sum = float(features.get("txn_sum_1min", 0))

    input_data = np.array([[txn_count, txn_sum]], dtype=np.float32)

    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)

    prediction = int(outputs[0][0])

    return {
        "prediction": prediction
    }
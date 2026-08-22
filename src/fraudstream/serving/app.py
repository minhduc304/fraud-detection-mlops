import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, Info, generate_latest

from fraudstream.features.schema import FEATURE_COLUMNS, RawTransaction
from fraudstream.features.transforms import build_features
from fraudstream.serving.model_loader import ModelLoader

PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds", "Time spent scoring a /predict request"
)
PREDICTION_SCORE = Histogram("fraud_prediction_score", "Predicted fraud score")
FEATURE_VALUE = Histogram(
    "fraud_feature_value", "Feature value seen at prediction time", ["feature"]
)
MODEL_VERSION_INFO = Info("fraud_model_version", "Model version currently serving predictions")


def create_app(loader: ModelLoader | None = None) -> FastAPI:
    app = FastAPI()
    app.state.loader = loader or ModelLoader()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response) -> dict[str, str]:
        try:
            app.state.loader.model
        except RuntimeError:
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}

    @app.post("/predict")
    def predict(txn: RawTransaction) -> dict[str, float | str]:
        try:
            model = app.state.loader.model
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Model not loaded") from exc
        start = time.perf_counter()
        df = pd.DataFrame([txn.model_dump()])
        features = build_features(df)
        score = float(model.predict(features)[0])
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
        PREDICTION_SCORE.observe(score)
        for col in FEATURE_COLUMNS:
            FEATURE_VALUE.labels(feature=col).observe(float(features.iloc[0][col]))
        MODEL_VERSION_INFO.info({"version": str(app.state.loader.version)})
        return {"score": score, "model_version": app.state.loader.version}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()

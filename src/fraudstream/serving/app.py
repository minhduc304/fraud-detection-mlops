import pandas as pd
from fastapi import FastAPI, HTTPException, Response

from fraudstream.features.schema import RawTransaction
from fraudstream.features.transforms import build_features
from fraudstream.serving.model_loader import ModelLoader


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
        df = pd.DataFrame([txn.model_dump()])
        features = build_features(df)
        score = float(model.predict(features)[0])
        return {"score": score, "model_version": app.state.loader.version}

    return app


app = create_app()

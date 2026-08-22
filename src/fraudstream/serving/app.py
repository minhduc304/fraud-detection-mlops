from fastapi import FastAPI, Response

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

    return app


app = create_app()

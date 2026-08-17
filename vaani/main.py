import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .deps import PipelineDeps, build_deps
from .orchestrator import CallSession
from .settings import Settings, load_settings

_DEMO_DIR = Path(__file__).resolve().parent / "demo"


def create_app(settings: Optional[Settings] = None, deps: Optional[PipelineDeps] = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="TBB-VaaniAI", version="0.1.0")
    app.state.settings = settings
    app.state.deps = deps

    app.mount("/demo", StaticFiles(directory=_DEMO_DIR, html=True), name="demo")

    @app.get("/")
    async def root():
        return RedirectResponse("/demo/")

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": app.state.settings.__class__.__name__}

    @app.websocket("/ws/call")
    async def ws_call(
        websocket: WebSocket,
        call_id: Optional[str] = Query(default=None),
    ):
        await websocket.accept()
        deps = app.state.deps
        if deps is None:
            deps = build_deps(app.state.settings)
            app.state.deps = deps
        session = CallSession(
            websocket, deps, app.state.settings, call_id or uuid.uuid4().hex[:12]
        )
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await websocket.close(code=1000)
            except Exception:
                pass

    return app


app = create_app()
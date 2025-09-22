"""ASGI entrypoint that mounts the existing FastAPI `api` from main.py
and exposes Prometheus metrics at /metrics without modifying the project code.
"""
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from importlib import import_module

# Import existing FastAPI instance named 'api' from main.py
main = import_module("backend.main")
api = getattr(main, "api")

metrics_app = make_asgi_app()

app = Starlette(routes=[
    Mount("/metrics", app=metrics_app),
    Mount("/", app=api)
])

# Keep permissive CORS (compose/ingress friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

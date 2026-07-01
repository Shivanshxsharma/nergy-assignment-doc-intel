# Makes backend/ a proper Python package.
# Required so `uvicorn backend.api:app` works from the project root
# without relying on sys.path tricks at module-load time.

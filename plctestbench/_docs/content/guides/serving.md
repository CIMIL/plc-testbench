# Serving documentation in an application

The package can build a complete static site from an installed distribution. This keeps documentation deployment separate from your service’s routing, authentication, TLS, and caching policy.

## Build or preview locally

```bash
pip install "plctestbench[docs]"
plctestbench-docs build /var/cache/plctestbench-docs --strict --profile embedded
plctestbench-docs serve --port 8000 --profile embedded
```

The default profile is `standalone`. Use `embedded` for application integrations; it replaces the package landing page and excludes standalone-only getting-started and repository navigation. `serve` binds to `127.0.0.1` by default and is intended for local preview. It builds the site once; run it again after source changes.

## Build from Python

```python
from pathlib import Path
from plctestbench.docs import build_docs

site_dir = build_docs(
    Path("/var/cache/plctestbench-docs"),
    strict=True,
    profile="embedded",
)
```

The return value is the directory containing `index.html` and all site assets. Build during deployment or application startup, then let the host framework serve that directory.

## FastAPI or Starlette

```python
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from plctestbench.docs import build_docs

app = FastAPI()
site_dir = build_docs("/var/cache/plctestbench-docs", profile="embedded")
app.mount("/plctestbench-docs", StaticFiles(directory=site_dir, html=True))
```

## Flask

```python
from flask import Flask, send_from_directory
from plctestbench.docs import build_docs

app = Flask(__name__)
site_dir = build_docs("/var/cache/plctestbench-docs", profile="embedded")

@app.get("/plctestbench-docs/")
@app.get("/plctestbench-docs/<path:path>")
def plctestbench_docs(path=""):
    return send_from_directory(site_dir, path or "index.html")
```

These snippets are examples only. The package intentionally does not depend on either framework.

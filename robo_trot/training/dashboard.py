from __future__ import annotations

import html
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


MEDIA_KEYS = (
    ("vx00", "vx00"),
    ("vx03", "vx03"),
    ("vx06", "vx06"),
    ("yaw_left", "yaw_left"),
    ("yaw_right", "yaw_right"),
)


def render_dashboard_html(run_dir: str | Path) -> str:
    """Render the static dashboard application shell.

    Runtime metrics are fetched from JSON API endpoints by browser-side code.
    """
    root = Path(run_dir)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BC Training Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{_dashboard_css()}</style>
</head>
<body>
  <header class="page-header">
    <div>
      <p class="eyebrow">Training Operations</p>
      <h1>Behavior Cloning Training</h1>
      <p id="run-path" class="run-path">{html.escape(str(root))}</p>
    </div>
    <button id="refresh-button" class="run-button" type="button">Refresh</button>
  </header>
  <main id="dashboard-root" class="dashboard-root">
    <section class="panel"><div class="waiting">Loading dashboard data...</div></section>
  </main>
  <script>{_dashboard_js()}</script>
</body>
</html>"""


def build_dashboard_payload(run_dir: str | Path) -> dict[str, Any]:
    """Return all dashboard data needed by the client application.

    This is also useful for tests and programmatic dashboard inspection.
    """
    root = Path(run_dir)
    mlp_train = _read_jsonl(root / "mlp" / "metrics.jsonl")
    txl_train = _read_jsonl(root / "txl" / "metrics.jsonl")
    eval_rows = _read_jsonl(root / "eval" / "metrics.jsonl")
    return {
        "run_dir": str(root),
        "cache_token": _cache_token(root),
        "summary": build_summary(root),
        "train_metrics": {"mlp": mlp_train, "txl": txl_train},
        "eval_metrics": eval_rows,
        "gifs": build_gif_index(root, eval_rows),
        "raw_links": _raw_links(),
    }


def build_summary(run_dir: str | Path) -> dict[str, Any]:
    """Return latest train and eval rows for both model families.

    Rows are kept as dictionaries so the browser can choose its display format.
    """
    root = Path(run_dir)
    eval_rows = _read_jsonl(root / "eval" / "metrics.jsonl")
    return {
        "models": {
            "mlp": {
                "train": _latest_jsonl(root / "mlp" / "metrics.jsonl"),
                "eval": _latest_eval(eval_rows, "mlp"),
            },
            "txl": {
                "train": _latest_jsonl(root / "txl" / "metrics.jsonl"),
                "eval": _latest_eval(eval_rows, "txl"),
            },
        }
    }


def build_gif_index(run_dir: str | Path, rows: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return evaluator media metadata grouped by model and checkpoint.

    Each entry includes a browser URL served by `/files/...`.
    """
    root = Path(run_dir)
    eval_rows = _dedupe_eval_rows(rows if rows is not None else _read_jsonl(root / "eval" / "metrics.jsonl"))
    result: dict[str, list[dict[str, Any]]] = {"mlp": [], "txl": []}
    token = _cache_token(root)
    for row in eval_rows:
        model = _row_model(row)
        if model not in result:
            result[model] = []
        gifs = {
            label: _media_payload(root, path, token)
            for label, path in _all_media_paths_for_row(root, row)
            if path is not None
        }
        result[model].append(
            {
                "model_type": model,
                "checkpoint_update": int(row.get("checkpoint_update", row.get("update", -1))),
                "wall_time": row.get("wall_time"),
                "eval_reward_mean": row.get("eval_reward_mean"),
                "eval_fall_rate": row.get("eval_fall_rate"),
                "eval_survival_seconds_mean": row.get("eval_survival_seconds_mean"),
                "dataset_eval_action_mse": row.get("dataset_eval_action_mse"),
                "gifs": gifs,
            }
        )
    for model_rows in result.values():
        model_rows.sort(key=lambda item: int(item.get("checkpoint_update", -1)))
    return result


def serve_dashboard(run_dir: str | Path, host: str = "0.0.0.0", port: int = 8002) -> None:
    """Serve dashboard HTML, JSON APIs, and static run files.

    The backend reads JSONL metrics on each request so manual refreshes show live data.
    """
    root = Path(run_dir).resolve()
    handler_class = _make_handler(root)
    server = ThreadingHTTPServer((str(host), int(port)), handler_class)
    server.serve_forever()


def _make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one run directory.

    API routes return JSON; `/files/` serves media, metrics, and checkpoint folders.
    """

    class DashboardHandler(BaseHTTPRequestHandler):
        """HTTP handler for the dashboard app and run-directory APIs.

        Path traversal outside the run directory is rejected for file routes.
        """

        def do_GET(self) -> None:
            """Serve the dashboard shell, JSON API, or one static file.

            Query parameters are parsed only for API endpoints that need them.
            """
            parsed = urlparse(self.path)
            if parsed.path in {"", "/"}:
                _send_bytes(self, render_dashboard_html(root).encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/dashboard":
                _send_json(self, build_dashboard_payload(root))
                return
            if parsed.path == "/api/summary":
                _send_json(self, build_summary(root))
                return
            if parsed.path == "/api/train-metrics":
                _send_json(self, _api_train_metrics(root, parsed.query))
                return
            if parsed.path == "/api/eval-metrics":
                _send_json(self, _read_jsonl(root / "eval" / "metrics.jsonl"))
                return
            if parsed.path == "/api/gifs":
                _send_json(self, build_gif_index(root))
                return
            if parsed.path.startswith("/files/"):
                _serve_file(self, root, parsed.path.removeprefix("/files/"))
                return
            self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress default access logs.

            Manual refreshes and media requests otherwise make stderr noisy.
            """
            del format, args

    return DashboardHandler


def _api_train_metrics(root: Path, query: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Return train metrics for one model or both models.

    The optional `model` query parameter accepts `mlp` or `txl`.
    """
    params = parse_qs(query)
    model = params.get("model", [""])[0]
    if model in {"mlp", "txl"}:
        return _read_jsonl(root / model / "metrics.jsonl")
    return {
        "mlp": _read_jsonl(root / "mlp" / "metrics.jsonl"),
        "txl": _read_jsonl(root / "txl" / "metrics.jsonl"),
    }


def _raw_links() -> dict[str, str]:
    """Return stable raw file links used by the browser app.

    Links are relative to the dashboard server root.
    """
    return {
        "mlp_metrics": "/files/mlp/metrics.jsonl",
        "txl_metrics": "/files/txl/metrics.jsonl",
        "eval_metrics": "/files/eval/metrics.jsonl",
        "mlp_latest": "/files/mlp/checkpoints/latest/",
        "txl_latest": "/files/txl/checkpoints/latest/",
    }


def _latest_eval(rows: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    """Return the latest eval row for a model type.

    Rows are sorted by checkpoint update and append order, preserving backfills.
    """
    matches = [row for row in rows if _row_model(row) == model]
    if not matches:
        return None
    return sorted(enumerate(matches), key=lambda item: (int(item[1].get("checkpoint_update", -1)), item[0]))[-1][1]


def _latest_jsonl(path: Path) -> dict[str, Any] | None:
    """Return the newest metric JSON object from a JSONL file.

    Rows with update counters are sorted so checkpoint resumes do not look stale.
    """
    rows = _read_jsonl(path)
    if not rows:
        return None
    if any("update" in row for row in rows):
        return sorted(enumerate(rows), key=lambda item: (int(item[1].get("update", -1)), item[0]))[-1][1]
    return rows[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL metric rows from disk.

    Missing and partially written files return the parseable rows available.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _dedupe_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one eval row per model/checkpoint pair.

    Later rows win so dataset-loss backfills are reflected in API payloads.
    """
    indexed: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        model = _row_model(row)
        update = int(row.get("checkpoint_update", row.get("update", -1)))
        if model and update >= 0:
            indexed[(model, update)] = (index, row)
    return [
        item[1]
        for _, item in sorted(indexed.items(), key=lambda pair: (pair[0][1], pair[0][0], pair[1][0]))
    ]


def _row_model(row: dict[str, Any]) -> str:
    """Return the normalized model key stored in an eval metric row.

    Older rows may use `model`; current rows use `model_type`.
    """
    return str(row.get("model_type", row.get("model", "")))


def _all_media_paths_for_row(root: Path, row: dict[str, Any]) -> list[tuple[str, Path | None]]:
    """Return all rollout media paths recorded for one eval row.

    The filesystem fallback scans only the checkpoint directory for that row.
    """
    paths: dict[str, Path | None] = {}
    for field in ("gif_paths", "media_paths"):
        raw_paths = row.get(field)
        if isinstance(raw_paths, dict):
            for key, value in raw_paths.items():
                candidate = root / str(value)
                label = str(key)
                if candidate.exists():
                    paths[label] = candidate
                else:
                    paths.setdefault(label, None)
    if not paths:
        model = _row_model(row)
        update = int(row.get("checkpoint_update", row.get("update", -1)))
        for step_dir in (
            root / "eval" / "media" / f"step_{update:09d}",
            root / "eval" / "gifs" / f"step_{update:09d}",
        ):
            if model and step_dir.exists():
                for suffix in ("*.mp4", "*.webm", "*.webp", "*.gif"):
                    for candidate in sorted(step_dir.glob(f"{model}_{suffix}")):
                        label = candidate.stem.removeprefix(f"{model}_")
                        paths.setdefault(label, candidate)
    return sorted(paths.items(), key=lambda item: _gif_sort_key(item[0]))


def _media_payload(root: Path, path: Path, token: str) -> dict[str, Any]:
    """Return browser metadata for one media path.

    Invalid paths are marked missing instead of raising during API generation.
    """
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return {"exists": False, "url": "", "path": path.as_posix()}
    suffix = path.suffix.lower()
    return {
        "exists": path.exists(),
        "path": rel,
        "url": f"/files/{rel}?v={token}",
        "kind": "video" if suffix in {".mp4", ".webm"} else "image",
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def _gif_sort_key(label: str) -> tuple[int, str]:
    """Sort known commands first and preserve unknown eval commands by name.

    New evaluator commands show up after the fixed command set.
    """
    known_order = {key: index for index, (key, _) in enumerate(MEDIA_KEYS)}
    return (known_order.get(label, len(known_order)), label)


def _cache_token(root: Path) -> str:
    """Return a cache-busting token derived from metric and media mtimes.

    The value changes whenever JSONL metrics or media outputs change.
    """
    latest = 0
    for pattern in ("**/*.jsonl", "**/*.gif", "**/*.mp4", "**/*.webm", "**/*.webp"):
        for path in root.glob(pattern):
            try:
                latest = max(latest, int(path.stat().st_mtime_ns))
            except OSError:
                continue
    return str(latest)


def _serve_file(handler: BaseHTTPRequestHandler, root: Path, relative_url: str) -> None:
    """Serve one file or directory listing below the run directory.

    Attempts to escape the run root are rejected with a 404 response.
    """
    relative = Path(unquote(relative_url))
    target = (root / relative).resolve()
    if root not in target.parents and target != root:
        handler.send_error(404)
        return
    if target.is_dir():
        listing = "\n".join(
            f'<li><a href="{html.escape(child.name)}">{html.escape(child.name)}</a></li>'
            for child in sorted(target.iterdir())
        )
        _send_bytes(handler, f"<ul>{listing}</ul>".encode("utf-8"), "text/html; charset=utf-8")
        return
    if not target.exists():
        handler.send_error(404)
        return
    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    _send_bytes(handler, target.read_bytes(), content_type)


def _send_json(handler: BaseHTTPRequestHandler, payload: Any) -> None:
    """Write a JSON HTTP response.

    JSON responses are UTF-8 and sorted only by insertion order from Python dicts.
    """
    _send_bytes(handler, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")


def _send_bytes(handler: BaseHTTPRequestHandler, body: bytes, content_type: str) -> None:
    """Write an HTTP 200 response with a byte body.

    The helper centralizes headers for HTML, JSON, JSONL, and media responses.
    """
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _dashboard_css() -> str:
    """Return CSS for the client-rendered dashboard shell.

    The style keeps the page dense and operational rather than decorative.
    """
    return """
:root {
  color-scheme: light;
  --bg: #f3f5f7;
  --surface: #ffffff;
  --surface-alt: #f8fafb;
  --ink: #172026;
  --muted: #5c6873;
  --line: #d9e0e6;
  --line-strong: #b8c2cc;
  --accent: #0f6f68;
  --warn: #875400;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}
.page-header {
  padding: 20px 28px 16px;
  border-bottom: 1px solid var(--line-strong);
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  background: var(--surface);
}
h1 { font-size: 24px; line-height: 1.2; margin: 0; font-weight: 700; letter-spacing: 0; }
h2, h3 { letter-spacing: 0; }
.eyebrow { margin: 0 0 5px; color: var(--accent); font-size: 11px; font-weight: 750; text-transform: uppercase; }
.run-path { margin: 7px 0 0; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
.run-button {
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  padding: 7px 12px;
  background: var(--surface-alt);
  color: var(--ink);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}
.dashboard-root {
  width: min(100%, 1680px);
  margin: 0 auto;
  padding: 20px 28px 34px;
  display: grid;
  gap: 16px;
}
.panel, .model-panel, .history-column {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  overflow: hidden;
}
.section-heading, .panel-heading, .column-heading {
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--surface);
}
.section-heading, .panel-heading {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}
.section-heading h2, .panel-heading h2, .column-heading h3 {
  margin: 0;
  font-size: 18px;
  line-height: 1.2;
}
.section-heading p, .panel-heading p, .column-heading p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}
.step-chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 3px 8px;
  background: var(--surface-alt);
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}
.status-grid, .charts-grid, .model-columns, .history-columns {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  padding: 14px;
  background: var(--surface-alt);
}
.status-card, .chart-card {
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--surface);
  overflow: hidden;
}
.status-title, .chart-title {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
}
.status-title h3, .chart-title h3 { margin: 0; font-size: 14px; line-height: 1.2; }
.status-facts {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
}
.fact { padding: 10px 12px; border-right: 1px solid var(--line); }
.fact:last-child { border-right: 0; }
.fact-label { display: block; color: var(--muted); font-size: 11px; line-height: 1.2; text-transform: uppercase; }
.fact-value { display: block; margin-top: 4px; font-size: 15px; font-weight: 700; overflow-wrap: anywhere; }
.plot { width: 100%; height: 300px; }
.plot-fallback { padding: 60px 12px; min-height: 200px; color: var(--warn); text-align: center; font-size: 13px; }
.metrics-table { width: 100%; border-collapse: collapse; font-size: 13px; background: var(--surface); }
.metrics-table th, .metrics-table td {
  text-align: left;
  padding: 7px 10px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
.metrics-table th { color: var(--muted); font-weight: 600; width: 48%; }
.block-heading { padding: 12px 14px 8px; border-top: 1px solid var(--line); background: var(--surface-alt); }
.block-heading h3 { margin: 0; font-size: 14px; line-height: 1.2; }
.block-heading p { margin: 4px 0 0; color: var(--muted); font-size: 12px; line-height: 1.35; }
.gifs, .history-gifs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  padding: 12px;
  background: var(--surface-alt);
}
.history-gifs { grid-template-columns: repeat(5, minmax(120px, 1fr)); padding: 10px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; min-height: 120px; }
figcaption { padding: 7px 8px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--line); }
img, video { display: block; width: 100%; height: auto; min-height: 90px; object-fit: contain; background: #eef2f4; }
.history-list { display: grid; gap: 12px; padding: 12px; }
.eval-card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.eval-title {
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  font-weight: 650;
}
.eval-meta { color: var(--muted); font-weight: 500; }
.links { padding: 10px 12px 14px; display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; background: var(--surface); border-top: 1px solid var(--line); }
a { color: var(--accent); text-decoration: none; font-weight: 650; }
a:hover { text-decoration: underline; }
.waiting { padding: 26px 8px; min-height: 90px; color: var(--warn); display: flex; align-items: center; justify-content: center; text-align: center; }
@media (max-width: 900px) {
  .page-header, .section-heading, .panel-heading { align-items: flex-start; flex-direction: column; }
  .dashboard-root { padding: 16px; }
  .status-grid, .charts-grid, .model-columns, .history-columns, .status-facts, .gifs, .history-gifs { grid-template-columns: 1fr; }
}
"""


def _dashboard_js() -> str:
    """Return browser-side dashboard code.

    The client fetches JSON APIs and uses Plotly for zoomable/pannable charts.
    """
    return r"""
const modelNames = {mlp: "MLP", txl: "TXL"};
const modelColors = {mlp: "#1d6fb8", txl: "#a34b2f"};
const rawLinks = {
  mlp: [["training metrics", "/files/mlp/metrics.jsonl"], ["latest checkpoint", "/files/mlp/checkpoints/latest/"], ["best val loss", "/files/mlp/checkpoints/best_val_loss/"], ["best eval reward", "/files/mlp/checkpoints/best_eval_reward/"]],
  txl: [["training metrics", "/files/txl/metrics.jsonl"], ["latest checkpoint", "/files/txl/checkpoints/latest/"], ["best val loss", "/files/txl/checkpoints/best_val_loss/"], ["best eval reward", "/files/txl/checkpoints/best_eval_reward/"]]
};

function sci(value) {
  if (value === null || value === undefined || value === "") return "pending";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return number.toExponential(4).replace(/e([+-])(\d)$/, (_match, sign, digit) => `e${sign}${digit.padStart(2, "0")}`);
}

function integerLabel(value) {
  if (value === null || value === undefined || value === "") return "pending";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return String(Math.trunc(number));
}

function stepLabel(value) {
  if (value === null || value === undefined || value === "") return "pending";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return String(Math.trunc(number)).padStart(9, "0");
}

function timeLabel(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "";
  return new Date(number * 1000).toISOString().replace("T", " ").replace(".000Z", " UTC");
}

function stepBadge(evalRow, trainRow) {
  const update = evalRow?.checkpoint_update ?? trainRow?.update;
  const stamp = timeLabel(evalRow?.wall_time);
  return `step ${stepLabel(update)}${stamp ? " · " + stamp : ""}`;
}

function metricRows(trainRow, evalRow) {
  return [
    ["checkpoint update", integerLabel(evalRow?.checkpoint_update ?? trainRow?.update)],
    ["train loss", sci(trainRow?.train_loss)],
    ["val loss", sci(trainRow?.val_loss)],
    ["action MSE", sci(trainRow?.action_mse)],
    ["action L1", sci(trainRow?.action_l1)],
    ["clip fraction", sci(trainRow?.action_clip_fraction)],
    ["dataset eval split", evalRow?.dataset_eval_split ?? "pending"],
    ["test action MSE", sci(evalRow?.dataset_eval_action_mse)],
    ["test action L1", sci(evalRow?.dataset_eval_action_l1)],
    ["test clip fraction", sci(evalRow?.dataset_eval_action_clip_fraction)],
    ["test label values", integerLabel(evalRow?.dataset_eval_values)],
    ["eval reward mean", sci(evalRow?.eval_reward_mean)],
    ["survival seconds", sci(evalRow?.eval_survival_seconds_mean)],
    ["fall rate", sci(evalRow?.eval_fall_rate)],
    ["forward velocity", sci(evalRow?.eval_forward_velocity_mean)],
    ["yaw response", sci(evalRow?.eval_yaw_rate_mean)],
    ["foot slip", sci(evalRow?.eval_foot_slip_mean)],
    ["samples seen", integerLabel(trainRow?.samples_seen)]
  ];
}

function htmlEscape(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
}

async function fetchJson(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

async function loadDashboard() {
  const [summary, trainMetrics, evalMetrics, gifs] = await Promise.all([
    fetchJson("/api/summary"),
    fetchJson("/api/train-metrics"),
    fetchJson("/api/eval-metrics"),
    fetchJson("/api/gifs")
  ]);
  return {summary, trainMetrics, evalMetrics, gifs};
}

function renderStatus(models) {
  return `<section class="panel">
    <div class="section-heading"><div><h2>Run Status</h2><p>Latest train and eval rows from API data.</p></div><span class="step-chip">manual reload</span></div>
    <div class="status-grid">${["mlp", "txl"].map((model) => statusCard(model, models[model])).join("")}</div>
  </section>`;
}

function statusCard(model, rows) {
  const train = rows?.train || {};
  const evalRow = rows?.eval || {};
  const facts = [
    ["train update", integerLabel(train.update)],
    ["eval step", integerLabel(evalRow.checkpoint_update)],
    ["train loss", sci(train.train_loss)],
    ["test action MSE", sci(evalRow.dataset_eval_action_mse)],
    ["eval reward", sci(evalRow.eval_reward_mean)]
  ];
  return `<article class="status-card">
    <div class="status-title"><h3>${modelNames[model]}</h3><span class="step-chip">${htmlEscape(stepBadge(evalRow, train))}</span></div>
    <div class="status-facts">${facts.map(([label, value]) => `<div class="fact"><span class="fact-label">${label}</span><span class="fact-value">${value}</span></div>`).join("")}</div>
  </article>`;
}

function renderCharts(trainMetrics, evalMetrics) {
  return `<section class="panel">
    <div class="section-heading"><div><h2>Curves</h2><p>Plotly widgets support zoom, pan, box select, autoscale, and image export.</p></div><span class="step-chip">interactive</span></div>
    <div class="charts-grid">
      ${chartCard("Training action MSE", "plot-train-loss")}
      ${chartCard("Validation action MSE", "plot-val-loss")}
      ${chartCard("MuJoCo eval reward", "plot-reward")}
      ${chartCard("Held-out dataset action MSE", "plot-test-mse")}
    </div>
  </section>`;
}

function chartCard(title, id) {
  return `<article class="chart-card"><div class="chart-title"><h3>${title}</h3><span class="step-chip">zoom / pan</span></div><div id="${id}" class="plot"></div></article>`;
}

function plotCharts(trainMetrics, evalMetrics) {
  if (!window.Plotly) {
    document.querySelectorAll(".plot").forEach((node) => node.innerHTML = '<div class="plot-fallback">Plotly failed to load. Check network access to the Plotly CDN.</div>');
    return;
  }
  plotLoss("plot-train-loss", trainMetrics, "train_loss", "Training action MSE");
  plotLoss("plot-val-loss", trainMetrics, "val_loss", "Validation action MSE");
  plotEval("plot-reward", evalMetrics, "eval_reward_mean", "MuJoCo eval reward");
  plotEval("plot-test-mse", evalMetrics, "dataset_eval_action_mse", "Held-out dataset action MSE");
}

function plotLoss(id, trainMetrics, key, title) {
  const traces = ["mlp", "txl"].map((model) => ({
    x: (trainMetrics[model] || []).map((row) => row.update),
    y: (trainMetrics[model] || []).map((row) => row[key]),
    name: `${modelNames[model]} ${key === "train_loss" ? "train" : "val"}`,
    type: "scatter",
    mode: "lines",
    line: {color: modelColors[model], width: 2}
  }));
  drawPlot(id, traces, title, "update", "normalized action MSE");
}

function plotEval(id, evalMetrics, key, title) {
  const deduped = dedupeEvalRows(evalMetrics);
  const traces = ["mlp", "txl"].map((model) => {
    const rows = deduped.filter((row) => row.model_type === model && row[key] !== undefined);
    return {
      x: rows.map((row) => row.checkpoint_update),
      y: rows.map((row) => row[key]),
      name: modelNames[model],
      type: "scatter",
      mode: "lines+markers",
      line: {color: modelColors[model], width: 2},
      marker: {size: 5}
    };
  });
  drawPlot(id, traces, title, "checkpoint update", key.includes("reward") ? "reward" : "normalized action MSE");
}

function drawPlot(id, traces, title, xTitle, yTitle) {
  const layout = {
    title: {text: title, font: {size: 13}},
    margin: {l: 54, r: 18, t: 38, b: 42},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "x unified",
    xaxis: {title: xTitle, showgrid: true, gridcolor: "#d9e0e6"},
    yaxis: {title: yTitle, showgrid: true, gridcolor: "#d9e0e6", tickformat: ".2e"},
    legend: {orientation: "h", y: 1.18, x: 0}
  };
  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  };
  Plotly.newPlot(id, traces.filter((trace) => trace.x.length), layout, config);
}

function dedupeEvalRows(rows) {
  const indexed = new Map();
  rows.forEach((row, index) => {
    const model = row.model_type || row.model || "";
    const update = row.checkpoint_update ?? row.update;
    if (!model || update === undefined) return;
    indexed.set(`${model}:${update}`, {...row, model_type: model, _index: index});
  });
  return Array.from(indexed.values()).sort((a, b) => (a.checkpoint_update - b.checkpoint_update) || a.model_type.localeCompare(b.model_type));
}

function renderModels(models, gifs) {
  return `<section class="panel">
    <div class="section-heading"><div><h2>Policy Metrics</h2><p>Latest supervised action-label diagnostics and checkpoint eval metrics.</p></div></div>
    <div class="model-columns">${["mlp", "txl"].map((model) => modelPanel(model, models[model], gifs[model] || [])).join("")}</div>
  </section>`;
}

function modelPanel(model, rows, gifRows) {
  const train = rows?.train || {};
  const evalRow = rows?.eval || {};
  const latestGifs = gifRows.length ? gifRows[gifRows.length - 1].gifs : {};
  return `<section id="${model}-panel" class="model-panel">
    <div class="panel-heading"><div><h2>${modelNames[model]}</h2><p>${model === "mlp" ? "Feed-forward policy baseline" : "Transformer-XL sequence policy"}</p></div><span class="step-chip">${htmlEscape(stepBadge(evalRow, train))}</span></div>
    ${metricsTable(train, evalRow)}
    <div class="block-heading"><h3>Latest Eval Media</h3><p>Looping MuJoCo rollouts for the latest evaluated checkpoint.</p></div>
    <div id="${model}-gif-gallery" class="gifs">${gifTiles(latestGifs)}</div>
    <nav id="${model}-raw-links" class="links">${rawLinks[model].map(([label, url]) => `<a href="${url}">${label}</a>`).join("")}<a href="/files/eval/metrics.jsonl">eval metrics</a></nav>
  </section>`;
}

function metricsTable(train, evalRow) {
  return `<table class="metrics-table">${metricRows(train, evalRow).map(([label, value]) => `<tr><th>${htmlEscape(label)}</th><td>${htmlEscape(value)}</td></tr>`).join("")}</table>`;
}

function gifTiles(gifs) {
  const labels = ["vx00", "vx03", "vx06", "yaw_left", "yaw_right"];
  return labels.map((label) => {
    const media = gifs[label];
    const body = media?.url ? mediaElement(media, label) : '<div class="waiting">Waiting for media</div>';
    return `<figure><figcaption>${label}</figcaption>${body}</figure>`;
  }).join("");
}

function mediaElement(media, label) {
  if (media.kind === "video") {
    return `<video autoplay muted loop playsinline preload="auto"><source src="${media.url}" type="${htmlEscape(media.mime_type || "video/mp4")}"></video>`;
  }
  return `<img src="${media.url}" alt="${htmlEscape(label)}" loading="lazy">`;
}

function renderGifHistory(gifs) {
  return `<section class="panel">
    <div class="section-heading"><div><h2>All Eval Media</h2><p>Historical MuJoCo rollout media, separated by model family.</p></div></div>
    <div class="history-columns">${["mlp", "txl"].map((model) => historyColumn(model, gifs[model] || [])).join("")}</div>
  </section>`;
}

function historyColumn(model, rows) {
  const body = rows.length ? rows.map((row) => historyCard(model, row)).join("") : `<div class="waiting">Waiting for ${modelNames[model]} evaluated checkpoints and media</div>`;
  return `<article class="history-column"><div class="column-heading"><h3>${modelNames[model]} History</h3><p>${rows.length} evaluated checkpoints.</p></div><div id="${model}-history-gallery" class="history-list">${body}</div></article>`;
}

function historyCard(model, row) {
  const meta = `reward ${sci(row.eval_reward_mean)} · survival ${sci(row.eval_survival_seconds_mean)}s · fall ${sci(row.eval_fall_rate)}`;
  return `<article class="eval-card"><div class="eval-title"><span>${modelNames[model]} step ${stepLabel(row.checkpoint_update)}</span><span class="eval-meta">${meta}</span></div><div class="history-gifs">${Object.entries(row.gifs || {}).map(([label, media]) => `<figure><figcaption>${label}</figcaption>${mediaElement(media, label)}</figure>`).join("")}</div></article>`;
}

async function render() {
  const root = document.getElementById("dashboard-root");
  try {
    const data = await loadDashboard();
    root.innerHTML = renderStatus(data.summary.models) + renderModels(data.summary.models, data.gifs) + renderCharts(data.trainMetrics, data.evalMetrics) + renderGifHistory(data.gifs);
    plotCharts(data.trainMetrics, data.evalMetrics);
  } catch (error) {
    root.innerHTML = `<section class="panel"><div class="waiting">Dashboard data load failed: ${htmlEscape(error.message)}</div></section>`;
  }
}

document.getElementById("refresh-button").addEventListener("click", render);
render();
"""

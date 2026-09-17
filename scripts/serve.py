"""Restricted localhost demo server and real trained-model inference API."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits
from foresight.env import NavigationEnv, ROBOT_RADIUS
from foresight.model import HybridWorldModel
from foresight.planner import MPCPlanner, PhysicsPredictor
from foresight.residual import ResidualCalibratedModel

METHODS = ("adaptive", "fixed_5", "fixed_10", "fixed_16", "global_physics", "local_identification", "residual_calibrated")
SCENARIOS = ("open", "crossing", "slalom")
MAX_BODY_BYTES = 8192
SESSION_LIMIT = 16
SESSION_TTL_SECONDS = 30 * 60


class APIError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _empty_frame(state):
    return {"state": state, "action": [0, 0], "horizon": 0, "planning_ms": 0,
            "model_steps": 0, "disagreement": 0, "predicted_path": [], "candidate_paths": []}


def _finite_number(value, label):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise APIError(f"{label} must be a finite number")
    return float(value)


@dataclass
class LiveSession:
    env: NavigationEnv
    planner: MPCPlanner
    method: str
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False


class LiveApplication:
    """Immutable model and individually locked, bounded episode sessions."""
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.sessions: dict[str, LiveSession] = {}
        self.sessions_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.model = None
        self.load_error = None
        self.budget = 4500
        self.threshold = 0.003
        self.global_damping = 1.0
        try:
            summary = json.loads((self.root / "artifacts/day0/summary.json").read_text(encoding="utf-8"))
            self.budget = int(summary["config"]["budget"])
            self.threshold = float(summary["calibration"]["threshold"])
            self.global_damping = float(summary["training"]["global_damping_fit"])
            checkpoint = self.root / "artifacts/day0/model.npz"
            if not checkpoint.exists():
                checkpoint = self.root / "checkpoints/day0_model.npz"
            self.model = HybridWorldModel.load(checkpoint)
        except (OSError, ValueError, KeyError, TypeError):
            self.load_error = "Run the day-zero experiment to create the trained checkpoint and summary."

    def status(self):
        return {"ready": self.model is not None, "model": "hybrid_world_model",
                "members": self.model.n_members if self.model is not None else 0,
                "methods": METHODS, "scenarios": SCENARIOS, "budget": self.budget,
                "uncertainty_threshold": self.threshold, "damping_scale_range": [0.5, 2.5],
                "error": self.load_error, "mode": "live_inference"}

    def _cleanup(self):
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        for key in [key for key, session in self.sessions.items() if session.touched < cutoff]:
            self.sessions.pop(key, None)

    def _session(self, body):
        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise APIError("session_id must be a nonempty string")
        with self.sessions_lock:
            self._cleanup()
            session = self.sessions.get(session_id)
            if session is None:
                raise APIError("Unknown or expired session; reset to start a new episode", 404)
            session.touched = time.monotonic()
            return session

    def reset(self, body):
        if self.model is None:
            raise APIError(self.load_error, 503)
        method = body.get("method", "adaptive")
        scenario = body.get("scenario", "crossing")
        seed = body.get("seed", 2026)
        scale = _finite_number(body.get("damping_scale", 1.0), "damping_scale")
        if method not in METHODS or scenario not in SCENARIOS:
            raise APIError("Unknown method or scenario")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 31 - 1:
            raise APIError("seed must be an integer between 0 and 2147483647")
        if not 0.5 <= scale <= 2.5:
            raise APIError("damping_scale must be between 0.5 and 2.5")
        predictor = (PhysicsPredictor(damping=self.global_damping, adaptive=method == "local_identification")
                     if method in {"global_physics", "local_identification"} else self.model)
        if method == "residual_calibrated":
            predictor = ResidualCalibratedModel(self.model)
        horizon = int(method.split("_")[-1]) if method.startswith("fixed_") else 10
        planner = MPCPlanner(predictor, horizon=horizon, adaptive=method == "adaptive",
                             budget=self.budget, seed=seed + 70000, uncertainty_threshold=self.threshold)
        env = NavigationEnv(seed=seed, scenario=scenario, damping_scale=scale)
        state = env.reset(seed=seed)
        session_id = secrets.token_urlsafe(24)
        with self.sessions_lock:
            self._cleanup()
            if len(self.sessions) >= SESSION_LIMIT:
                oldest = min(self.sessions, key=lambda key: self.sessions[key].touched)
                del self.sessions[oldest]
            self.sessions[session_id] = LiveSession(env=env, planner=planner, method=method)
        return {"session_id": session_id, "frame": _empty_frame(state), "done": False,
                "method": method, "seed": seed, "scenario": scenario, "damping_scale": scale}

    def step(self, body):
        session = self._session(body)
        with session.lock:
            if session.done:
                raise APIError("Episode ended; reset to start a new episode", 409)
            state = session.env.state
            # BLAS thread limits are process-global; serialize inference while scoped.
            with self.model_lock, threadpool_limits(limits=1):
                action, info = session.planner.act(state)
                next_state, _, terminated, truncated, terminal = session.env.step(action)
                session.planner.observe(state, action, next_state)
            session.done = bool(terminated or truncated)
            frame = {**info, "state": state, "action": action}
            frame["candidate_paths"] = info.get("candidate_paths", [])[:5]
            predictor = session.planner.predictor
            if hasattr(predictor, "current_scale"):
                # Calibration occurs after executing this action and is available
                # to the next decision; planner timing remains a separate metric.
                frame.update(learned_scale=float(predictor.current_scale),
                             scale_updates=int(predictor.updates),
                             calibration_ms=float(predictor.last_calibration_ms))
            return {"frame": frame, "next_state": next_state, "done": session.done,
                    "success": terminal["success"], "collision": terminal["collision"],
                    "timeout": terminal["timeout"], "step": session.env.steps,
                    "min_clearance": terminal["min_clearance"]}

    def goal(self, body):
        x, y = _finite_number(body.get("x"), "x"), _finite_number(body.get("y"), "y")
        if not (ROBOT_RADIUS < x < 10 - ROBOT_RADIUS and ROBOT_RADIUS < y < 10 - ROBOT_RADIUS):
            raise APIError("Goal must be inside the map with robot-radius clearance from walls")
        session = self._session(body)
        with session.lock:
            if session.done:
                raise APIError("Episode ended; reset before choosing another goal", 409)
            state = session.env.state
            state[4:6] = [x, y]
            session.env.state = state
            session.planner.reset()
            return {"session_id": body["session_id"], "frame": _empty_frame(state), "done": False}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app: LiveApplication, **kwargs):
        self.app = app
        super().__init__(*args, directory=str(app.root), **kwargs)

    def _reply(self, status, payload):
        encoded = json.dumps(_jsonable(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _local_api(self):
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            if host.hostname not in {"localhost", "127.0.0.1"} or host.port != self.server.server_port:
                raise APIError("API requests must use this localhost server", 403)
            origin_header = self.headers.get("Origin")
            if origin_header:
                origin = urlsplit(origin_header)
                if origin.scheme != "http" or origin.hostname not in {"localhost", "127.0.0.1"} or origin.port != self.server.server_port:
                    raise APIError("Cross-origin API requests are not allowed", 403)
        except ValueError:
            raise APIError("Invalid request origin", 403)

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise APIError("Transfer-Encoding is not supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError("Invalid Content-Length")
        if length < 1 or length > MAX_BODY_BYTES:
            raise APIError("JSON body must contain 1 to 8192 bytes", 413)
        if self.headers.get_content_type() != "application/json":
            raise APIError("Content-Type must be application/json", 415)
        try:
            body = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            pending = [body]
            while pending:
                item = pending.pop()
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError("nonfinite JSON number")
                if isinstance(item, dict):
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise APIError("Body must be valid finite JSON")
        if not isinstance(body, dict):
            raise APIError("JSON body must be an object")
        return body

    def do_POST(self):
        try:
            self._local_api()
            path = urlsplit(self.path).path
            routes = {"/api/reset": self.app.reset, "/api/step": self.app.step, "/api/goal": self.app.goal}
            if path not in routes:
                raise APIError("Unknown API endpoint", 404)
            self._reply(200, routes[path](self._body()))
        except APIError as error:
            self._reply(error.status, {"error": str(error)})
        except Exception:
            self._reply(500, {"error": "Inference failed; inspect the local server terminal"})
            raise

    def _static_target(self):
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
        except (ValueError, UnicodeError):
            return None
        if "\\" in path or "\x00" in path or ":" in path:
            return None
        parts = path.lstrip("/").split("/")
        if not parts or parts[0] not in {"web", "artifacts"} or any(part in {".", ".."} for part in parts):
            return None
        allowed = self.app.root / parts[0]
        candidate = self.app.root.joinpath(*parts)
        try:
            if not candidate.resolve().is_relative_to(allowed.resolve()):
                return None
            current = self.app.root
            for part in parts:
                current /= part
                if current.is_symlink():
                    return None
            if candidate.is_dir():
                for index_name in ("index.html", "index.htm"):
                    index = candidate / index_name
                    if index.is_symlink() or not index.resolve().is_relative_to(allowed.resolve()):
                        return None
        except OSError:
            return None
        return candidate

    def translate_path(self, path):
        target = self._static_target()
        return str(target if target is not None else self.app.root / "__not_available__")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            try:
                self._local_api()
                self._reply(200, self.app.status())
            except APIError as error:
                self._reply(error.status, {"error": str(error)})
            return
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/web/")
            self.end_headers()
            return
        if self._static_target() is None:
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if self._static_target() is None:
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def create_server(port=8765, root=ROOT):
    app = LiveApplication(root)
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, app=app))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f"Foresight Lab: http://127.0.0.1:{server.server_port}/web/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

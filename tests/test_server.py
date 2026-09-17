"""HTTP checks for localhost live inference and restricted file serving."""
import http.client
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from scripts.serve import APIError, LiveApplication, ROOT, create_server


class LiveServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        if isinstance(body, dict):
            body = json.dumps(body)
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        payload = json.loads(raw) if response.getheader("Content-Type", "").startswith("application/json") else raw
        connection.close()
        return status, payload

    def test_invalid_json_and_session_are_rejected(self):
        for body in ("[]", '{"seed":NaN}', '{"extra":[1e999]}', '{"seed":', "{}" * 5000):
            status, payload = self.request("POST", "/api/reset", body)
            self.assertIn(status, (400, 413))
            self.assertIn("error", payload)
        status, payload = self.request("POST", "/api/step", {"session_id": "missing"})
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_remote_origin_and_host_are_rejected(self):
        status, _ = self.request("GET", "/api/status", headers={"Origin": "https://example.com"})
        self.assertEqual(status, 403)
        status, _ = self.request("GET", "/api/status", headers={"Host": "example.com"})
        self.assertEqual(status, 403)

    def test_gated_resources_are_required_without_affecting_old_model(self):
        with tempfile.TemporaryDirectory() as directory:
            app = LiveApplication(directory)
            # A usable legacy model does not authorize silently falling back to
            # unmatched weights when the new gate metadata is unavailable.
            app.model = object()
            with self.assertRaises(APIError) as caught:
                app.reset({"method": "gated_calibrated"})
            self.assertEqual(caught.exception.status, 503)
            self.assertIn("matching calibration", str(caught.exception))

    def test_private_files_and_encoded_traversal_are_not_served(self):
        for path in ("/README.md", "/.git/config", "/web/%2e%2e/README.md",
                     "/web/%2e%2e%5cREADME.md", "/web/%252e%252e/README.md",
                     "/artifacts/../checkpoints/day0_model.npz"):
            status, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
        status, _ = self.request("GET", "/")
        self.assertEqual(status, 302)

    def test_symlinks_do_not_escape_static_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            web = root / "web"
            web.mkdir()
            secret = root / "private.txt"
            secret.write_text("not public", encoding="utf-8")
            try:
                (web / "index.html").symlink_to(secret)
                (web / "linked.txt").symlink_to(secret)
            except OSError:
                self.skipTest("Host does not allow creating test symlinks")
            server = create_server(port=0, root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for path in ("/web/", "/web/linked.txt"):
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                    connection.request("GET", path)
                    response = connection.getresponse()
                    response.read()
                    self.assertEqual(response.status, 404)
                    connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    @unittest.skipUnless(((ROOT / "checkpoints/day0_model.npz").exists() or (ROOT / "artifacts/day0/model.npz").exists()) and
                         (ROOT / "artifacts/day0/summary.json").exists(), "Trained checkpoint unavailable")
    def test_actual_reset_step_goal_and_input_validation(self):
        status, metadata = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(metadata["ready"])
        for invalid in ({"seed": True}, {"damping_scale": 3.0}, {"method": "unknown"}, {"seed": -1}):
            status, _ = self.request("POST", "/api/reset", invalid)
            self.assertEqual(status, 400)
        status, episode = self.request("POST", "/api/reset", {"method": "adaptive", "seed": 41,
                                                                "scenario": "crossing", "damping_scale": 1.0})
        self.assertEqual(status, 200)
        self.assertEqual(len(episode["frame"]["state"]), 18)
        session_id = episode["session_id"]
        status, step = self.request("POST", "/api/step", {"session_id": session_id})
        self.assertEqual(status, 200)
        self.assertEqual(step["step"], 1)
        self.assertEqual(step["frame"]["state"], episode["frame"]["state"])
        self.assertEqual(len(step["next_state"]), 18)
        self.assertGreater(step["frame"]["model_steps"], 0)
        self.assertGreater(len(step["frame"]["predicted_path"]), 1)
        status, goal = self.request("POST", "/api/goal", {"session_id": session_id, "x": 8, "y": 3})
        self.assertEqual(status, 200)
        self.assertEqual(goal["frame"]["state"][4:6], [8.0, 3.0])
        status, _ = self.request("POST", "/api/goal", {"session_id": session_id, "x": 0, "y": 3})
        self.assertEqual(status, 400)

    @unittest.skipUnless((ROOT / "artifacts/day0/model.npz").exists() and
                         (ROOT / "artifacts/day0/summary.json").exists(), "Trained checkpoint unavailable")
    def test_residual_calibration_is_live_and_session_local(self):
        status, episode = self.request("POST", "/api/reset", {"method": "residual_calibrated", "seed": 41,
                                                                "scenario": "open", "damping_scale": 1.7})
        self.assertEqual(status, 200)
        for step_number in (1, 2):
            status, step = self.request("POST", "/api/step", {"session_id": episode["session_id"]})
            self.assertEqual(status, 200)
            self.assertEqual(step["step"], step_number)
            self.assertEqual(step["frame"]["horizon"], 10)
            self.assertTrue(math.isfinite(step["frame"]["learned_scale"]))
            self.assertTrue(math.isfinite(step["frame"]["calibration_ms"]))
            self.assertGreaterEqual(step["frame"]["calibration_ms"], 0)
        self.assertGreater(step["frame"]["scale_updates"], 0)
        status, another = self.request("POST", "/api/reset", {"method": "residual_calibrated", "seed": 41,
                                                                "scenario": "open", "damping_scale": 1.7})
        self.assertEqual(status, 200)
        _, first_step = self.request("POST", "/api/step", {"session_id": another["session_id"]})
        self.assertEqual(first_step["frame"]["learned_scale"], 1.0)
        self.assertEqual(first_step["frame"]["scale_updates"], 0)

    @unittest.skipUnless((ROOT / "artifacts/day0/model.npz").exists() and
                         (ROOT / "artifacts/day0/summary.json").exists(), "Trained checkpoint unavailable")
    def test_mid_episode_dynamics_changes_physics_without_informing_planner(self):
        app = self.server.RequestHandlerClass.keywords["app"]
        session_ids = []
        for _ in range(2):
            status, episode = self.request("POST", "/api/reset", {
                "method": "fixed_10", "seed": 41, "scenario": "open", "damping_scale": 1.0})
            self.assertEqual(status, 200)
            session_ids.append(episode["session_id"])
            # Start with a visible, unsaturated velocity to make the physics
            # difference identifiable even though both planners see one state.
            env = app.sessions[episode["session_id"]].env
            state = env.state
            state[:4] = [2.0, 2.0, 0.4, 0.1]
            env.state = state
        session = app.sessions[session_ids[1]]
        before = session.env.state.copy()
        status, response = self.request("POST", "/api/dynamics", {
            "session_id": session_ids[1], "damping_scale": 1.7})
        self.assertEqual(status, 200)
        self.assertEqual(response["damping_scale"], 1.7)
        np.testing.assert_array_equal(session.env.state, before)
        self.assertEqual(session.env.steps, 0)
        self.assertIs(session.planner.predictor, app.model)
        results = [self.request("POST", "/api/step", {"session_id": session_id})
                   for session_id in session_ids]
        for status, _ in results:
            self.assertEqual(status, 200)
        nominal, changed = (result[1] for result in results)
        self.assertEqual(nominal["frame"]["state"], changed["frame"]["state"])
        self.assertEqual(nominal["frame"]["action"], changed["frame"]["action"])
        self.assertEqual(nominal["frame"]["predicted_path"], changed["frame"]["predicted_path"])
        self.assertNotEqual(nominal["next_state"][:4], changed["next_state"][:4])
        for invalid in (None, True, 0.49, 2.51):
            status, _ = self.request("POST", "/api/dynamics", {
                "session_id": session_ids[1], "damping_scale": invalid})
            self.assertEqual(status, 400)
        session.done = True
        status, _ = self.request("POST", "/api/dynamics", {
            "session_id": session_ids[1], "damping_scale": 1.0})
        self.assertEqual(status, 409)

    @unittest.skipUnless((ROOT / "artifacts/day0/model.npz").exists() and
                         (ROOT / "artifacts/day0/summary.json").exists(), "Trained checkpoint unavailable")
    def test_legacy_methods_remain_available(self):
        for method in ("adaptive", "fixed_5", "fixed_10", "fixed_16",
                       "global_physics", "local_identification", "residual_calibrated"):
            with self.subTest(method=method):
                status, episode = self.request("POST", "/api/reset", {
                    "method": method, "seed": 91, "scenario": "open"})
                self.assertEqual(status, 200)
                status, result = self.request("POST", "/api/step", {"session_id": episode["session_id"]})
                self.assertEqual(status, 200)
                self.assertGreater(result["frame"]["model_steps"], 0)
                self.assertNotIn("gate_active", result["frame"])

    @unittest.skipUnless((ROOT / "artifacts/gated-study/calibration.json").exists(),
                         "Gated calibration artifacts unavailable")
    def test_gated_live_metadata_and_dynamics_preserve_calibration(self):
        status, metadata = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(metadata["gated_ready"])
        status, episode = self.request("POST", "/api/reset", {
            "method": "gated_calibrated", "seed": 41, "scenario": "open"})
        self.assertEqual(status, 200)
        self.assertEqual(episode["frame"]["model_training_seed"], metadata["gated_model_training_seed"])
        self.assertEqual(episode["frame"]["learned_scale"], 1.0)
        self.assertFalse(episode["frame"]["gate_active"])
        for _ in range(3):
            status, result = self.request("POST", "/api/step", {"session_id": episode["session_id"]})
            self.assertEqual(status, 200)
        for name in ("gate_score", "gate_threshold", "raw_scale", "learned_scale", "calibration_ms"):
            self.assertTrue(math.isfinite(result["frame"][name]))
        app = self.server.RequestHandlerClass.keywords["app"]
        session = app.sessions[episode["session_id"]]
        diagnostic_before = app._diagnostics(session)
        status, _ = self.request("POST", "/api/dynamics", {
            "session_id": episode["session_id"], "damping_scale": 1.7})
        self.assertEqual(status, 200)
        self.assertEqual(app._diagnostics(session), diagnostic_before)


if __name__ == "__main__":
    unittest.main()

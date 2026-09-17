"""HTTP checks for localhost live inference and restricted file serving."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from scripts.serve import ROOT, create_server


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


if __name__ == "__main__":
    unittest.main()

"""Serve only the demo and publishable artifacts on localhost."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            self.send_response(302)
            self.send_header('Location', '/web/')
            self.end_headers()
            return
        if '..' in path or not (path.startswith('/web/') or path.startswith('/artifacts/')):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        path = urlsplit(self.path).path
        if '..' in path or not (path.startswith('/web/') or path.startswith('/artifacts/')):
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(root)))
    print(f'Foresight Lab: http://127.0.0.1:{args.port}/web/', flush=True)
    server.serve_forever()

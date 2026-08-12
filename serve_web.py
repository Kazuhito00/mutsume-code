"""ブラウザデモをローカルで確認するためのサーバ。

    python serve_web.py [ポート]      # 既定 8000 -> http://localhost:8000

`web/` と `mutsume/` を `_site/` に集めて配信する（GitHub Actions と同じ構成）。
Windows の http.server は `.js` を text/plain で返して module script が
弾かれることがあるので、MIME を明示的に登録してから起動する。
"""

import functools
import http.server
import mimetypes
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(ROOT, "_site")

shutil.rmtree(SITE, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, "web"), SITE)
shutil.copytree(os.path.join(ROOT, "mutsume"), os.path.join(SITE, "mutsume"))

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=SITE)
print(f"serving {SITE} at http://localhost:{port}  (Ctrl+C で終了)")
http.server.HTTPServer(("", port), handler).serve_forever()

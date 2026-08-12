"""ブラウザ (Pyodide) から呼ぶヘルパ。JS とやり取りしやすいよう入出力を
JSON / base64 / RGBA バイト列に寄せてある。コアの mutsume は無改造で使う。
"""

import base64
import io
import json

import numpy as np
from PIL import Image

import mutsume

# 動画トラッキング用に前フレームで読めた姿勢を保持する
_hints = []


def do_encode(text, ecc, profile, palette, cell_size):
    """テキストをエンコードし {"png": base64, profile, palette, radius} を返す。"""
    kw = {"ecc": ecc, "palette": palette}
    if profile and profile != "auto":
        kw["profile"] = profile
    sym = mutsume.encode(text, **kw)
    buf = io.BytesIO()
    sym.to_image(cell_size=cell_size).save(buf, "PNG")
    return json.dumps({
        "png": base64.b64encode(buf.getvalue()).decode(),
        "profile": sym.profile, "palette": sym.palette, "radius": sym.radius,
    })


def _geom(r):
    d = {"text": r.text, "profile": r.profile, "palette": r.palette,
         "radius": r.radius, "ecc": r.ecc}
    g = r.geometry
    if g is not None:
        d["outline"] = [[float(x), float(y)] for x, y in g.outline]
        d["finders"] = [[float(x), float(y)] for x, y in g.finders]
        d["center"] = [float(g.center[0]), float(g.center[1])]
        d["corner0"] = [float(g.corner0[0]), float(g.corner0[1])]
        d["rotation"] = float(g.rotation_deg)
        d["mirrored"] = bool(g.mirrored)
    return d


def decode_rgba(buf, w, h, max_symbols=4, use_hints=False, hints_only=False):
    """canvas の RGBA バイト列を復号し、結果 (JSON 配列) を返す。

    use_hints=True で前フレームの姿勢を追従に使う (動画向け)。
    hints_only=True なら追従のみ (全探索を省いて速い)。座標は入力画像基準。
    """
    global _hints
    arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    img = Image.fromarray(arr)
    hints = _hints if (use_hints and _hints) else None
    ho = bool(hints_only and hints)
    results = mutsume.decode_all(img, max_symbols=max_symbols,
                                 hints=hints, hints_only=ho)
    _hints = [r.geometry for r in results if r.geometry]
    return json.dumps([_geom(r) for r in results])


def reset_hints():
    global _hints
    _hints = []

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


def _hex(h):
    """'#rrggbb' -> (r, g, b)。"""
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _kw(ecc, profile):
    kw = {"ecc": ecc}
    if profile and profile != "auto":
        kw["profile"] = profile
    return kw


def do_encode(text, ecc, profile, cell_size, dark_hex, light_hex,
              grid_lines=True, invert=False, dark_separator=True):
    """テキストをエンコードし {"ok": True, "png": base64, ...} を返す。

    容量超過などで生成できないときは {"ok": False, "error": 説明}（例外は投げない）。
    dark_hex / light_hex で暗いセル・明るいセルの色を指定する。invert=True で
    暗色と明色を入れ替える (色相は保ったまま明暗を反転する)。
    grid_lines は明るいセルの枠線、dark_separator は暗いセル同士を分ける枠線。
    """
    try:
        sym = mutsume.encode(text, **_kw(ecc, profile))
    except mutsume.MutsumeError as e:
        return json.dumps({"ok": False, "error": str(e)})
    dark, light = _hex(dark_hex), _hex(light_hex)
    if invert:
        dark, light = light, dark
    buf = io.BytesIO()
    sym.to_image(cell_size=cell_size, grid_lines=grid_lines,
                 dark_separator=dark_separator,
                 dark_rgb=dark, light_rgb=light).save(buf, "PNG")
    return json.dumps({
        "ok": True,
        "png": base64.b64encode(buf.getvalue()).decode(),
        "profile": sym.profile, "palette": sym.palette, "radius": sym.radius,
    })


def check_encode(text, ecc, profile):
    """現在の設定でエンコードできるかを調べる (色・反転は容量に無関係)。

    {"ok": True, "profile", "radius"} か {"ok": False, "error": 説明} を返す。
    画像は作らないので軽く、入力のたびに呼んでライブ判定できる。
    """
    if not text:
        return json.dumps({"ok": False, "error": "テキストを入力してください。"})
    try:
        sym = mutsume.encode(text, **_kw(ecc, profile))
        return json.dumps({"ok": True, "profile": sym.profile, "radius": sym.radius})
    except mutsume.MutsumeError as e:
        return json.dumps({"ok": False, "error": str(e)})


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

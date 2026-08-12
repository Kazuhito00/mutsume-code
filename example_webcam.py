"""Web カメラで mutsume-code を読み取るシンプルなデモ。

    venv\\Scripts\\python.exe example_webcam.py                 # カメラ 0 番
    venv\\Scripts\\python.exe example_webcam.py --camera 1
    venv\\Scripts\\python.exe example_webcam.py --video path/to/movie.mp4
    venv\\Scripts\\python.exe example_webcam.py --image encode_basic.png

前フレームで読めた位置を追従して高速化する (既定オン。--refresh で全探索の間隔を
調整、1 で毎フレーム全探索)。検出したコードの外形・ファインダ・向きを緑で描き、
その右上に読み取り結果を出す。左上に FPS・復号時間・検出数を表示。

キー: q / ESC で終了

必要なもの: venv\\Scripts\\python.exe -m pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    print("opencv-python が必要です:\n"
          "  venv\\Scripts\\python.exe -m pip install -r requirements.txt",
          file=sys.stderr)
    raise SystemExit(1)

from PIL import Image

import mutsume
from mutsume.render import ORIENTATION_ARROW_SHRINK

GREEN = (60, 220, 60)  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def decode_frame(frame, decode_width, max_symbols, hints=None, hints_only=False):
    """1 フレームを復号して (検出結果, 所要秒) を返す。

    復号前に decode_width まで縮小すると速く、実写ではモアレも減って
    検出率が上がる。座標は縮小画像基準で返るので、元フレームへ戻す。

    hints に前フレームの Geometry を渡すとトラッキング (検出を跳ばして
    前回の姿勢の近傍だけで復号) になり、1 フレームが数 ms で済む。hints は
    縮小画像基準なので、フレーム幅が一定なら次フレームへそのまま使える。
    """
    scale, work = 1.0, frame
    if decode_width and frame.shape[1] > decode_width:
        scale = decode_width / frame.shape[1]
        work = cv2.resize(frame, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))

    t0 = time.time()
    results = mutsume.decode_all(img, max_symbols=max_symbols,
                                 hints=hints, hints_only=hints_only)
    elapsed = time.time() - t0

    # scaled() は縮小画像基準の geometry を別オブジェクトへ写す。描画にはこれを
    # 使い、次フレームの hints には未スケールの res.geometry をそのまま渡す。
    back = 1.0 / scale
    found = [(r, r.geometry.scaled(back)) for r in results if r.geometry]
    return found, elapsed


def draw(frame, found, fps, elapsed):
    """外形・ファインダ・向き・テキストを緑で重ね描きする。"""
    for res, g in found:
        poly = np.array([[int(x), int(y)] for x, y in g.outline], np.int32)
        cv2.polylines(frame, [poly], True, GREEN, 2, cv2.LINE_AA)

        # ファインダ 3 点
        r = max(4, int(round(g.cell_size * 0.42)))
        for x, y in g.finders:
            cv2.circle(frame, (int(x), int(y)), r, GREEN, 2, cv2.LINE_AA)

        # 向き: 中心からコーナー 0 (シンボルの「正面」) へ矢印。
        # シグネチャ (micro は CRC) が回転 x 鏡映を解いているので向きは一意。
        cx, cy = int(g.center[0]), int(g.center[1])
        tx, ty = g.corner0
        k = ORIENTATION_ARROW_SHRINK
        cv2.arrowedLine(frame, (cx, cy),
                        (int(cx + (tx - cx) * k), int(cy + (ty - cy) * k)),
                        GREEN, 2, cv2.LINE_AA, tipLength=0.25)

        text = res.text if len(res.text) <= 40 else res.text[:39] + "…"
        x0, y0 = int(g.bbox[0]), int(g.bbox[1])
        cv2.putText(frame, text, (x0, y0 - 8), FONT, 0.6, GREEN, 2, cv2.LINE_AA)

    status = (f"FPS {fps:4.1f}   decode {elapsed * 1000:4.0f} ms"
              f"   detected {len(found)}   [q]quit")
    cv2.putText(frame, status, (8, 22), FONT, 0.6, GREEN, 2, cv2.LINE_AA)


def open_source(args):
    """(cap, still) を返す。--image のときは cap=None で静止画を繰り返す。"""
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"画像を開けません: {args.image}")
        return None, still
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera,
                               cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        raise SystemExit(f"開けません (camera={args.camera}, video={args.video})")
    return cap, None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="mutsume-code Web カメラデモ (シンプル)")
    p.add_argument("--camera", type=int, default=0, help="カメラ番号")
    p.add_argument("--video", help="動画ファイルから読む")
    p.add_argument("--image", help="静止画から読む (繰り返し表示)")
    p.add_argument("--decode-width", type=int, default=700,
                   help="復号前に縮小する横幅 (0 で無効)。小さいほど速い")
    p.add_argument("--max-symbols", type=int, default=2,
                   help="1 フレームで探すコードの最大数")
    p.add_argument("--refresh", type=int, default=10,
                   help="全探索を行う間隔 (フレーム数)。間のフレームは前回の"
                        "位置を追従するだけなので速い。1 で毎フレーム全探索")
    args = p.parse_args(argv)

    cap, still = open_source(args)
    fps, fps_t0, fps_n = 0.0, time.time(), 0
    total = 0
    hints: list = []  # 前フレームで読めた姿勢 (トラッキング用、既定オン)
    try:
        while True:
            if still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            total += 1
            # 全探索は refresh フレームごと。間のフレームは前回の位置を追従する
            # (新しく現れたコードは次の全探索で拾う)
            full = (not hints) or args.refresh <= 1 or (total % args.refresh == 1)
            found, elapsed = decode_frame(
                frame, args.decode_width or None, args.max_symbols,
                hints=hints or None, hints_only=not full)
            hints = [res.geometry for res, _ in found if res.geometry]

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 0.5:
                fps, fps_t0, fps_n = fps_n / (now - fps_t0), now, 0

            draw(frame, found, fps, elapsed)
            cv2.imshow("mutsume-code", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

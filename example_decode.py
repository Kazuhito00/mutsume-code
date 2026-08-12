"""mutsume-code を読み取る最小例。

    venv\\Scripts\\python.exe example_decode.py [画像パス]

引数を省くと example_encode.py が書き出す encode_basic.png を読む。
"""

import sys

import mutsume

path = sys.argv[1] if len(sys.argv) > 1 else "encode_basic.png"

# コードを 1 つ読む (profile / palette / 半径・向きは自動判別。読めなければ MutsumeError)
res = mutsume.decode(path)
print("text           :", res.text)
print("profile/palette:", res.profile, res.palette, "R =", res.radius)

g = res.geometry   # 画像上の位置・姿勢 (復号に使ったホモグラフィから求まる)
if g is not None:
    print("center         :", g.center)
    print("rotation       :", f"{g.rotation_deg:+.0f} deg")

# 1 枚に複数写っているときは decode_all ですべて返す
found = mutsume.decode_all(path, max_symbols=4)
print(f"\ndecode_all: {len(found)} 個")
for r in found:
    print(" -", r.text)

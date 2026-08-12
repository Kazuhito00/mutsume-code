"""mutsume-code でコードを生成する最小例。

    venv\\Scripts\\python.exe example_encode.py
"""

import mutsume

# 文字列をエンコード (profile / palette / 半径は自動で決まる)
sym = mutsume.encode("https://example.com/mutsume-code")
sym.save("encode_basic.png")   # 拡張子で PNG / SVG を自動判別
print(sym.to_text())           # コンソールで確認したいとき

# 主なオプション
mutsume.encode("IMPORTANT", ecc="H").save("encode_ecc_h.png")          # 誤り訂正を強く
mutsume.encode("123456789", profile="micro").save("encode_micro.png")  # 最小サイズ

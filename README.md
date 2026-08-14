# mutsume-code
六角格子の 2次元コードの PoC です。<br>
実用性皆無のジョークアプリのようなものなので、実用性を求める方はQRコードなどご利用ください。<br>

<img width="41%" alt="image" src="https://github.com/user-attachments/assets/d1e7daab-e805-4750-ab2a-0a675005bde0" />　<img width="52%" alt="image" src="https://github.com/user-attachments/assets/29cb750d-9719-4a2e-9795-b9266d793708" />

## 想定用途

- 何か他とは違ったデザインの2次元コードを付与したいとき
- SF作品の漫画とかゲーム内の世界観を補強する小道具
- etc

## セットアップ

```powershell
pip install -r requirements.txt
```

依存は `numpy` と `pillow` のみ。<br>
Web カメラデモを利用する際のみ `opencv-python` を利用。

## CLI

```powershell
# 生成
python -m mutsume encode "https://example.com" -o out.png --ecc Q

# 斜めから撮る前提なら robust プロファイル
python -m mutsume encode "..." -o out.png --profile robust

# 小さくしたいなら micro
python -m mutsume encode 123456789 -o out.png --profile micro

# 読み取り（プロファイル・向き・白黒反転は自動判別）
python -m mutsume decode out.png -v

# 画像上の位置を出す / 重ね描き画像を書き出す
python -m mutsume decode out.png --geometry
python -m mutsume decode out.png --overlay overlay.png

# サイズ別の容量表
python -m mutsume info --mode numeric
```

`-o out.svg` にすると SVG で出力する。

## Python API
```python
import mutsume

# 生成
sym = mutsume.encode("https://example.com", ecc="Q")          # profile= / palette= も可
sym.save("out.png", cell_size=18)   # 拡張子で PNG / SVG を自動判別
img = sym.to_image()                # PIL Image が欲しいとき
print(sym.to_text())                # ASCII アート

# 読み取り（ファイルパス / PIL.Image / Symbol を受け付ける）
res = mutsume.decode("out.png")
print(res.text, res.ecc, res.errors_corrected)

# 複数・トラッキング
results = mutsume.decode_all("photo.jpg", max_symbols=4)

# 何文字入るかの見積もり
mutsume.capacity(10)                  # -> バイト数
mutsume.capacity(10, mode="numeric")  # -> 数字の桁数
```

返り値は `Symbol`（生成）と `DecodeResult`（読み取り。`.text` / `.payload` /
`.geometry` を持つ）。読めなかったときは `MutsumeError`（`decode_all` は空リスト）。

`res.geometry` からファインダ中心・シンボル外形・任意セルの画素座標・傾き・
鏡映などが取れる（座標は入力画像そのままのピクセル単位）。

## 使用例スクリプト

```powershell
python example_encode.py                    # encode_*.png / .svg を生成
python example_decode.py encode_basic.png   # それを読む

# カメラ / 動画 / 静止画で読むシンプルなデモ（要 opencv-python）
python example_webcam.py                     # カメラ 0 番
python example_webcam.py --video path/to/movie.mp4
python example_webcam.py --image encode_basic.png
```

`example_webcam.py` は検出したコードの外形・ファインダ・向きを緑で描き、右上に
読み取り結果を出す。前フレームで読めた位置を追従して高速化する（既定オン、
`--refresh` で全探索の間隔を調整）。`q` / ESC で終了。

## ブラウザデモ（GitHub Pages）
[kazuhito00.github.io/mutsume-code/](https://kazuhito00.github.io/mutsume-code/)

Pyodide ベースのデモを用意しています。<br>
生成・画像からの読み取り・カメラ読み取りをブラウザ内（WASM）で実行する。<bR>
※コアの `mutsume` は無改造。numpy / pillow はPyodide が供給し、カメラは JS 側で取得するので opencv は不要

```powershell
# ローカル確認（web/ と mutsume/ を集めて配信。.js の MIME も正しく返す）
python serve_web.py       # http://localhost:8000
```

## テスト
```powershell
python -m unittest discover -s tests -v
```

## 構成

```
mutsume/
  rs.py       GF(256) と Reed-Solomon（消失訂正つき）
  bits.py     ビット列と文字モードのセグメント分割
  palette.py  カラーパレットと色判定
  layout.py   六角格子のジオメトリ、マーカー配置、螺旋順、マスク
  pose.py     姿勢（ホモグラフィ）と画像座標の取得 Geometry
  codec.py    符号語の組み立て / 分解、RS ブロック分割、マスク選択、向き探索
  render.py   PNG / SVG / ASCII 描画、位置の重ね描き
  detect.py   二値化、マーカー検出、ホモグラフィ推定、サンプリング
  cli.py      コマンドライン
tests/                単体テストと往復テスト（画像を tests/generated/ に生成）
example_encode.py     生成の最小例
example_decode.py     読み取りの最小例
example_webcam.py     カメラ / 動画 / 静止画で読むデモ（要 opencv-python）
web/                  ブラウザ (Pyodide) デモ。GitHub Pages で配信
.github/workflows/    Pages への自動デプロイ
```

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
mutsume-code is under [Apache-2.0 License](LICENSE).

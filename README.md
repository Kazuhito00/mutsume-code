# mutsume-code

六角格子の 2 次元コード（エンコーダ / デコーダ）の PoC。

蜂の巣状に隣接した六角セル、位置検出マーカー、Reed-Solomon 誤り訂正を備え、
**画像から回転・拡大縮小・鏡映・せん断・射影歪み・照明ムラを吸収して読み取る**。

## できること

- マーカープロファイル 3 種: `micro`（最小） / `compact`（標準） / `robust`（射影変換対応）
- カラーパレット 3 種: `mono` 1bit / `color4` 2bit / `color8` 3bit（容量 2〜3 倍）
- 文字モード: 数字 / 英数 / バイト。DP で最適セグメント分割
- Reed-Solomon ブロック分割 + インターリーブ、消失訂正（遮蔽への耐性）
- 適応的二値化（Sauvola）で照明ムラに対応
- 1 枚から複数コードを検出、動画では前フレームを追従
- 画像上の位置・姿勢（`Geometry`）の取得

## セットアップ

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

依存は `numpy` と `pillow` のみ（Reed-Solomon も画像処理も自前実装）。
Web カメラデモだけ `opencv-python` を使う。

## CLI

```powershell
# 生成
.\venv\Scripts\python.exe -m mutsume encode "https://example.com" -o out.png --ecc Q

# 斜めから撮る前提なら robust プロファイル
.\venv\Scripts\python.exe -m mutsume encode "..." -o out.png --profile robust

# 小さくしたいなら micro / 容量を稼ぎたいならカラー
.\venv\Scripts\python.exe -m mutsume encode 123456789 -o out.png --profile micro
.\venv\Scripts\python.exe -m mutsume encode "..." -o out.png --palette color4

# 読み取り（プロファイル / パレットは自動判別）
.\venv\Scripts\python.exe -m mutsume decode out.png -v

# 画像上の位置を出す / 重ね描き画像を書き出す
.\venv\Scripts\python.exe -m mutsume decode out.png --geometry
.\venv\Scripts\python.exe -m mutsume decode out.png --overlay overlay.png

# サイズ別の容量表
.\venv\Scripts\python.exe -m mutsume info --palette color4 --mode numeric
```

`-o out.svg` にすると SVG で出力する。

## Python API

公開 API は **4 関数**だけ。

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
.\venv\Scripts\python.exe example_encode.py                    # encode_*.png / .svg を生成
.\venv\Scripts\python.exe example_decode.py encode_basic.png   # それを読む

# カメラ / 動画 / 静止画で読むシンプルなデモ（要 opencv-python）
.\venv\Scripts\python.exe example_webcam.py                     # カメラ 0 番
.\venv\Scripts\python.exe example_webcam.py --video path/to/movie.mp4
.\venv\Scripts\python.exe example_webcam.py --image encode_basic.png
```

`example_webcam.py` は検出したコードの外形・ファインダ・向きを緑で描き、右上に
読み取り結果を出す。前フレームで読めた位置を追従して高速化する（既定オン、
`--refresh` で全探索の間隔を調整）。`q` / ESC で終了。

## テスト

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Reed-Solomon の誤り / 消失訂正、格子の不変条件、文字モード、カラーパレット、
全 12 向きの復元、PNG 経由の往復（回転 / 縮小 / ぼかし / JPEG / せん断 / 射影 /
照明ムラ / 遮蔽）を検証する。`test_roundtrip_images.py` は各プロファイル・
パレット・文字モード（枠線あり / なしを含む）で往復を検証し、生成した
PNG / SVG を `tests/generated/` に残す。

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
```

## 現状の限界

- 射影歪みは傾き 15% 程度まで（視野角 45 度超は不可）
- `compact` / `micro` は射影変換に非対応 — 斜め撮影用途なら `robust`
- カラーはレンダリング画像での検証のみ。実印刷 → 撮影の色再現は未検証

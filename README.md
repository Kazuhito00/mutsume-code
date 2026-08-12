# mutsume-code

六角格子の 2 次元コード（エンコーダ / デコーダ）の PoC。

蜂の巣状に隣接した六角セル、位置検出マーカー、Reed-Solomon 誤り訂正を備え、
**画像から回転・拡大縮小・鏡映・せん断・射影歪み・照明ムラを吸収して読み取る**
ところまで実装してある。

設計の詳細は [DESIGN.md](DESIGN.md) を参照。

## できること

- マーカープロファイル 3 種: `micro`（Micro QR 相当の最小） / `compact`（標準） / `robust`（射影変換対応）
- カラーパレット 3 種: `mono` 1bit / `color4` 2bit / `color8` 3bit（容量 2〜3 倍）
- 文字モード: 数字 3.33bit/文字、英数 5.5bit/文字、バイト。DP で最適セグメント分割
- Reed-Solomon ブロック分割 + インターリーブ（最大 1365 バイト）
- 消失訂正 (erasure) — 遮蔽への耐性が約 2 倍
- 適応的二値化 (Sauvola) — 照明ムラのある画像に対応

## セットアップ

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

依存は `numpy` と `pillow` のみ（Reed-Solomon も画像処理も自前実装）。

## CLI

```powershell
# 生成
.\venv\Scripts\python.exe -m mutsume encode "https://example.com" -o out.png --ecc Q

# 斜めから撮る前提なら robust プロファイル
.\venv\Scripts\python.exe -m mutsume encode "..." -o out.png --profile robust

# とにかく小さくしたいなら micro (Micro QR 相当)
.\venv\Scripts\python.exe -m mutsume encode 123456789 -o out.png --profile micro

# 容量を稼ぎたいならカラー
.\venv\Scripts\python.exe -m mutsume encode "..." -o out.png --palette color4

# 読み取り (プロファイル / パレットは自動判別)
.\venv\Scripts\python.exe -m mutsume decode out.png -v

# サイズ別の容量表
.\venv\Scripts\python.exe -m mutsume info --palette color4 --mode numeric
```

`-o out.svg` にすると SVG で出力する。`--show` で ASCII 表示も出る。

### 見た目のオプション

```powershell
# 白セルの黒枠線を消す (黒セル同士を分ける白セパレータは残る)
.\venv\Scripts\python.exe -m mutsume encode "..." -o out.png --no-grid-lines

# 逆に、黒セル同士の白セパレータをやめてベタ塗りに戻す
.\venv\Scripts\python.exe -m mutsume encode "..." -o out.png --no-separator
```

| 境界 | 既定 | `--no-grid-lines` |
|---|---|---|
| 黒 ↔ 黒 | 白の細線 | 白の細線（そのまま） |
| 白 ↔ 白 / 白 ↔ 黒 | 黒の細線 | なし |

`--no-grid-lines` だと白セルが背景に溶け、黒いセルの塊だけが浮かぶ見え方になる。
検出は「黒に囲まれた白」「白に囲まれた黒」を収縮 + 連結成分で見ているので、
**枠線の有無で読み取り耐性は変わらない**（実測でも同一）。

## 容量

### micro プロファイル（最小フォーマット）

小さいシンボルでは固定オーバーヘッドが容量を食い尽くす。`micro` は Micro QR と
同じ発想で、そこを徹底的に削ったフォーマット。

- 機能セルは**ロケータ 21 セルのみ**（シグネチャ・フォーマット領域なし）
- ヘッダは**モード 2bit + 個数 6bit の 1 バイトだけ**（仕様版・長さ 2B を廃止）
- CRC-16 は符号語末尾に固定配置（位置が既知なので長さフィールド不要）
- ECC レベルは固定（約 25%、下限 2 バイト）。マスク・向き・半径は
  **復号側で総当たりし CRC で確定**するので、シンボルに書かない

| R | 幅(セル) | セル数 | データ | 数字 | 英数 | バイト | 参考: compact の数字 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **5** | 11 | 91 | 8B | **7** | 4 | 3 | （不可） |
| **6** | 13 | 127 | 13B | **14** | 8 | 6 | （不可） |
| **7** | 15 | 169 | 18B | **26** | 16 | 11 | 9 |
| **8** | 17 | 217 | 24B | **36** | 21 | 15 | 23 |

compact が成立しない R=5/6 でも使え、R=7 では数字が **9 → 26 桁（2.9 倍）**になる。

代償: ECC レベルを選べない、セグメント分割なし（1 モードのみ）、
復号が総当たりぶん重い（0.15〜0.3 秒）、パレットは mono 固定。
R=9 以上は compact のほうが入るので `micro` は R=5..8 のみ。

```powershell
.\venv\Scripts\python.exe -m mutsume encode 123456789 --profile micro -o m.png
.\venv\Scripts\python.exe -m mutsume info --profile micro --mode numeric
```

### compact プロファイル（標準）

機能セルは **45 個で固定**
（ロケータ 7×3=21 + シグネチャ 6 + フォーマット 18）で半径によらないため、
大きくするほど相対オーバーヘッドは下がる。

「幅」は横方向のセル数（= 2R+1）。1 セル 1mm で刷るなら R=14 で約 29mm 角。
「日本語」は UTF-8 3 バイト換算。

### 白黒 (mono) / ECC=M

| R | 幅(セル) | セル数 | 数字 | 英数 | バイト | 日本語 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 15 | 169 | 9 | 5 | 3 | 1 |
| **8** | 17 | 217 | **23** | 14 | 9 | 3 |
| 9 | 19 | 271 | 35 | 21 | 14 | 4 |
| **10** | 21 | 331 | **47** | 28 | 19 | 6 |
| 11 | 23 | 397 | 64 | 38 | 26 | 8 |
| **12** | 25 | 469 | **81** | 49 | 33 | 11 |
| 13 | 27 | 547 | 93 | 56 | 38 | 12 |
| **14** | 29 | 631 | **114** | 69 | 47 | 15 |
| 15 | 31 | 721 | 136 | 82 | 56 | 18 |
| 16 | 33 | 817 | 155 | 94 | 64 | 21 |
| 18 | 37 | 1027 | 203 | 123 | 84 | 28 |
| **20** | 41 | 1261 | **256** | 155 | 106 | 35 |
| 24 | 49 | 1801 | 378 | 229 | 157 | 52 |
| 28 | 57 | 2437 | 522 | 316 | 217 | 72 |
| 32 | 65 | 3169 | 683 | 414 | 284 | 94 |
| 36 | 73 | 3997 | 870 | 527 | 362 | 120 |
| **40** | 81 | 4921 | **1079** | 654 | 449 | 149 |

ECC=Q にすると各値は約 2 割減、ECC=L なら約 1 割増。
`python -m mutsume info --mode numeric` で任意の条件の表が出せる。

### 実際のデータで必要な最小サイズ (ECC=M)

| 内容 | 最小 R | 幅 |
|---|---:|---:|
| 数字 20 桁 | 8 | 17 セル |
| 英数 10 文字 | 8 | 17 セル |
| 英数 25 文字 | 10 | 21 セル |
| URL 28 文字 | 12 | 25 セル |
| 日本語 10 文字 | 12 | 25 セル |
| URL 50 文字 | 15 | 31 セル |
| 日本語 30 文字 | 19 | 39 セル |

数字・英数は専用モードが効くので、同じサイズにバイトモードの
**2.4 倍 / 1.45 倍**入る。

### 実用下限は R=8

最小の R=7 はデータ領域が 15B しかなく、ECC を引くと数字 9 桁 / 英数 5 文字 /
**バイト 3 文字**。ECC=H では 1 バイトまで落ちるので実質使えない。
compact なら **R=8 以上**、それより小さくしたいなら `micro` を使う。

なお RS パリティには下限 4 バイトを設けてあるため、R=7 では
**L / M / Q がすべて同じ（nsym=4, 2 バイトまで訂正）**になる。差が出るのは H だけ。

R=7 に数字 9 桁を入れた例（`samples/r7_num_M_grid.png` / `r7_num_M_nogrid.png`）、
および micro で同じ 9 桁を R=6 に入れた例（`samples/micro_6_9.png`）:

```powershell
.\venv\Scripts\python.exe -m mutsume encode 123456789 --radius 7 -o r7.png
.\venv\Scripts\python.exe -m mutsume encode 123456789 --profile micro -o m.png
```

いずれも回転 33 度・鏡映・縮小 30%・ぼかし σ=2 で復号を確認済み。

### 同じサイズでもっと入れたい場合

カラーパレットを使うと、セル数を増やさずに容量が 2〜3 倍になる（ECC=M、数字/英数/バイト）:

| R | mono | color4 (2bit) | color8 (3bit) |
|---:|---|---|---|
| 8 | 23 / 14 / 9 | 61 / 37 / 25 | 97 / 59 / 40 |
| 10 | 47 / 28 / 19 | 109 / 66 / 45 | 177 / 107 / 73 |
| 12 | 81 / 49 / 33 | 174 / 105 / 72 | 268 / 162 / 111 |
| 14 | 114 / 69 / 47 | 246 / 149 / 102 | 378 / 229 / 157 |
| 20 | 256 / 155 / 106 | 529 / 321 / 220 | 803 / 486 / 334 |

たとえば **R=10 の color4（幅 21 セル）で英数 66 文字**は、白黒 R=16（幅 33 セル）と
ほぼ同じ容量が面積約 1/2.5 に収まる。

### robust プロファイルとの差

`robust`（射影変換対応）にすると機能セルが 45 → 93 に増えるが、
増えるのは固定 48 セルぶんだけなので、大きいシンボルほど差は縮む。

| R | compact (バイト) | robust (バイト) | 差 |
|---:|---:|---:|---:|
| 10 | 19 | 15 | −21% |
| 20 | 106 | 102 | −4% |
| 40 | 449 | 445 | −1% |

## 複数のコードをまとめて読む

1 枚に複数写っている場合は `decode_all` ですべて返す。
読めたコードの外形を除外領域に足しながら探索を繰り返す。

```python
import mutsume

for res in mutsume.decode_all("photo.jpg", max_symbols=4):
    print(res.text, res.geometry.center, res.geometry.bbox)
```

2 つめ以降も作業サイズ・二値化を絞らずに探索し直す。1 枚に写ったコードは
大きさも見え方も違い、当たる条件がそれぞれ違うため
（実測: 条件を絞らない 42% / サイズ固定 5% / 二値化固定 9%）。
そのかわり二値化とロケータ候補の抽出は条件ごとに 1 回だけ行ってキャッシュする。

動画では `hints` に前フレームの `geometry` を渡すとトラッキングになる:

```python
prev = []
for frame in video:
    results = mutsume.decode_all(frame, max_symbols=2, hints=prev,
                                hints_only=bool(prev))  # 定期的に False を挟む
    prev = [r.geometry for r in results]
```

## 実写映像での検出

`sample.mp4`（1920x1080・ノート PC の画面を手持ちで撮影、コード 2 つが同時に写る）
での実測。斜め・モアレ・ブレ・照明ムラを含む。

| | 当初 | 現在 |
|---|---:|---:|
| 1 つ以上を検出したフレーム | 10.5% | **99.0%** |
| 2 つとも検出したフレーム | 0% | **42.1%** |
| 1 コード検出の所要時間 (フル解像度) | 874 ms | **270 ms** |
| 2 コード検出の所要時間 (作業幅 600px) | - | **110 ms** |

2 つとも検出できるのは 42% だが、これは多くのフレームで片方が画面外・
極端に斜め・ボケているため。両方まともに写っているフレームではほぼ確実に取れる。

効いたのは次の 2 点。

**1. ロケータ三つ組の足切り**（10.5% → 26%）。
ロケータ 3 点の相互距離は必ず `√3 × リング数 × セルピッチ` になるので、
`辺長 / (√3 × ピッチ)` がありうる半径にならない組は捨てる。
画面を撮ると UI の白い部分が大量に候補になり、それらが作る
「大きいがピッチと辺長が全く釣り合わない三角形」が上位を占めていた。

**2. 作業画像サイズの階段**（26% → 96%）。
縮小すると UI の細かい模様やモアレが平滑化されてノイズ候補が激減し、
**検出率が上がる**。同じ映像で 1400px 24% → 600px 66% → 420px 91%。
小さい方から順に試し、当たったら判明した半径で最大サイズを引き直して
座標精度を戻す（粗く検出 → 精密化）。

## 画像上の位置を取得する

復号時に求めた姿勢（ホモグラフィ）をそのまま返すので、
**ファインダの中心・シンボルの外形・任意セルの画素座標**が取れる。
座標はすべて**入力画像そのままのピクセル単位**（内部の 1400px 縮小は打ち消し済み）。

```python
import mutsume

res = mutsume.decode("photo.jpg")
g = res.geometry

g.finders        # ファインダ 3 点  [(x, y), ...]
g.alignments     # アライメント 3 点 (robust のみ)
g.outline        # シンボル外形の 6 頂点
g.center         # 中心
g.bbox           # (x0, y0, x1, y1)
g.cell_size      # 隣接セル中心間の距離 (px)
g.rotation_deg   # 傾き (コーナー 0 方向の角度)
g.corner0        # コーナー 0 = シンボルの「正面」の外形頂点
g.mirrored       # 鏡映 (裏返し) に見えているか
g.perspective    # 射影補正が効いたか
g.homography     # 3x3 行列 (モデル座標 -> 画像座標)

g.cell_center((0, 0))     # 任意セルの中心
g.cell_polygon((0, 0))    # 任意セルの六角形 6 頂点
g.cell_centers()          # 全セルの中心 dict
g.to_dict()               # JSON にできる素の dict

# 確認用の重ね描き (デバッグ機能なのでサブモジュールから)
from mutsume.render import draw_geometry
draw_geometry(img, g, "overlay.png")
```

CLI からも:

```powershell
.\venv\Scripts\python.exe -m mutsume decode out.png --geometry
.\venv\Scripts\python.exe -m mutsume decode out.png --overlay overlay.png
```

出力例（`--geometry`）:

```
radius         14
profile        robust
finders        [[874.55, 442.34], [313.55, 118.32], [313.57, 766.31]]
alignments     [[687.56, 118.32], [126.57, 442.32], [687.54, 766.33]]
outline        [[952.47, 442.34], [726.52, 50.81], ...]
center         [500.55, 442.33]
bbox           [48.66, 50.81, 952.47, 833.82]
cell_size      31.17
rotation_deg   0.0
perspective    True
```

`--overlay` は外形を緑、ファインダを赤、アライメントを青、中心を橙の十字で
重ねた画像を書き出す（`samples/overlay_*.png`）。

実測精度: レンダリング時の既知座標との誤差は**サブピクセル〜1.7px**
（1.7px は入力が 1400px を超えて内部縮小が入った場合）。
回転・鏡映・縮小・射影歪みのいずれでも同等。

## Python API

公開 API は **4 関数**だけ。

```python
import mutsume

# 生成
sym = mutsume.encode("https://example.com/mutsume-code", ecc="Q")
sym = mutsume.encode(data, profile="robust", palette="color4")  # オプション
sym.save("out.png", cell_size=18)   # 拡張子で PNG / SVG を自動判別
img = sym.to_image()                # PIL Image が欲しいとき
print(sym.to_text())                # ASCII アート

# 読み取り (ファイルパス / PIL.Image / Symbol を受け付ける)
res = mutsume.decode("out.png")
print(res.text, res.ecc, res.errors_corrected, res.perspective)

# 複数・トラッキング
results = mutsume.decode_all("photo.jpg", max_symbols=4)

# 何文字入るかの見積もり
mutsume.capacity(10)                  # -> 19 (バイト)
mutsume.capacity(10, mode="numeric")  # -> 47 (数字)
```

返り値の型は `Symbol`（生成結果）と `DecodeResult`（読み取り結果、
`.text` / `.payload` / `.geometry` を持つ）。読めなかったときは
`MutsumeError`（`decode_all` は空リスト）。

内部機能（描画オプションの詳細・検出のレポート・格子演算など）は
サブモジュールから使える: `mutsume.render` / `mutsume.detect` / `mutsume.codec` /
`mutsume.layout`。

## 構成

```
mutsume/
  rs.py       GF(256) と Reed-Solomon (BM + Chien + Forney、消失訂正つき)
  bits.py     ビット列と文字モード (数字 / 英数 / バイト) のセグメント分割
  palette.py  カラーパレットと色判定
  layout.py   六角格子のジオメトリ、マーカー配置、螺旋順、マスク
  pose.py     姿勢 (ホモグラフィ) と画像座標の取得 Geometry
  codec.py    符号語の組み立て / 分解、RS ブロック分割、マスク選択、向き探索
  render.py   PNG / SVG / ASCII 描画、位置の重ね描き
  detect.py   二値化、マーカー検出、ホモグラフィ推定、サンプリング
  cli.py      コマンドライン
tests/                単体テスト。test_roundtrip_images.py は各パターンで
                      encode -> 保存 -> decode を検証し tests/generated/ に画像を出す
example_encode.py     生成の最小例
example_decode.py     読み取りの最小例
example_webcam.py     カメラ / 動画 / 静止画で読むシンプルなデモ (要 opencv-python)
```

## テスト

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Reed-Solomon の誤り / 消失訂正、格子の不変条件、RS ブロック分割、文字モード、
カラーパレット、論理層の往復、全 12 向きの復元、PNG 経由の往復（回転 / 縮小 /
ぼかし / JPEG / せん断 / 射影 / 照明ムラ / 部分遮蔽）を検証する。

`test_roundtrip_images.py` は各プロファイル・パレット・ECC・文字モード（白セルの
枠線あり / なしを含む）で encode → 保存 → decode の往復を検証し、生成した
PNG / SVG を `tests/generated/` に残す（目視でも確認できる）。

## Web カメラ / 静止画で読む

`example_webcam.py` はカメラ・動画・静止画から読み取るシンプルなデモ
（トラッキングなし、毎フレーム全探索）。検出したコードの外形を緑で囲み、
その右上に読み取り結果を出す。左上に FPS・復号時間・検出数を表示する。

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe example_webcam.py                       # カメラ 0 番
.\venv\Scripts\python.exe example_webcam.py --camera 1
.\venv\Scripts\python.exe example_webcam.py --video sample.mp4
.\venv\Scripts\python.exe example_webcam.py --image encode_basic.png
```

`q` / ESC で終了。`--decode-width`（既定 700）を小さくすると速く、実写では
モアレも減って検出率が上がる。`--max-symbols`（既定 2）は 1 フレームで探す
コードの最大数。1 フレーム復号してから同じフレームに描く同期処理なので、
オーバーレイは必ず「今見えている画」と一致する。opencv-python が必要。

## 使用例

`example_encode.py` / `example_decode.py` が最小の使い方を示す。

```powershell
.\venv\Scripts\python.exe example_encode.py                    # encode_*.png / .svg を生成
.\venv\Scripts\python.exe example_decode.py encode_basic.png
```

各種変形への耐性（実測）:

| 条件 | compact/mono 枠線あり | compact/mono 枠線なし | robust/mono 枠線なし | robust/color4 |
|---|---|---|---|---|
| 回転・鏡映・縮小 22% | OK | OK | OK | OK |
| ぼかし σ=2 / ノイズ σ=25 / JPEG q35 | OK | OK | OK | OK |
| せん断 0.2 | OK | OK | OK | OK |
| 射影 傾き 8% / 15% | 不可 | 不可 | **OK** | **OK** |
| 照明ムラ（0.35 / 0.15 倍） | OK | OK | OK | OK |
| 合計 | 14/16 | 14/16 | **16/16** | **16/16** |

中心の遮蔽は ECC=H で面積比 25% まで復号できる（消失訂正あり）。

## 現状の限界

- 射影歪みは傾き 15% 程度まで（視野角 45 度超は不可）
- `compact` は射影変換に対応しない — 斜め撮影用途なら `robust`
- カラーはレンダリング画像での検証のみ。実印刷 → 撮影の色再現は未検証

詳細と改善方針は [DESIGN.md](DESIGN.md) の「既知の限界 / 今後の拡張」に記載。

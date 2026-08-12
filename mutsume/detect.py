"""画像からのシンボル検出とサンプリング。

パイプライン:
  1. グレースケール化 + Otsu 二値化
  2. 白領域の連結成分ラベリング (ラン長ベースの union-find)
  3. 「黒に完全に囲まれた 1 セル大の白領域」= ロケータ中心の候補を抽出
  4. 候補 3 点が正三角形をなす組を探し、辺長からシンボル半径 R を推定
  5. 3 点対応 (6 通り) からアフィン変換を決め、全セル中心をサンプリング
  6. シグネチャ一致を優先しつつ CRC まで通る候補を採用
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image

from .codec import DecodeResult, MutsumeError, decode_grid, read_format
from .palette import DEFAULT_PALETTE, classify_batch
from .pose import Geometry, build_geometry, project, project_xy
from .layout import (
    Axial,
    CELL_PITCH,
    DEFAULT_PROFILE,
    PROFILES,
    get_layout,
    max_radius,
    min_radius,
    to_cartesian,
)

MAX_WORK_SIZE = 1400  # 作業画像の最大辺
MAX_TRIPLES = 16  # 検証するロケータ三つ組の上限
# 作業画像の長辺の階段。実写では縮小するとノイズ候補が激減して検出率が上がる。
WORK_SIZE_LADDER = (900, 600, 420)
EXCLUDE_MARGIN = 1.15  # 検出済み領域を除外する半径の余裕
# 2 つめ以降のシンボルを探すときの上限 (三つ組の数, 姿勢の試行回数)
EXTRA_SYMBOL_BUDGET = (6, 6)
POSE_TRY_MIN = 0.75  # 復号まで試す姿勢スコアの下限
TRACK_BINARIZERS = ("otsu", "sauvola8")  # トラッキング時に試す二値化
TRACK_SPAN = 1.0  # ファインダを探し直す範囲 (セルピッチ単位)
MAX_FULL_ATTEMPTS = 12  # 全セルサンプリング + 復号を試す姿勢候補の上限
MAX_MICRO_ATTEMPTS = 48  # micro は候補に順位が付かないので多めに試す
MAX_EARLY_ATTEMPTS = 3  # ほぼ確実な姿勢を即試す回数の上限
MAX_ERASURE_POSES = 3  # 消失訂正の再試行までする姿勢候補の数 (上位のみ)
# 幾何推定した半径のまわりで試す幅。ピッチ推定の相対誤差ぶんを見込む。
RADIUS_WINDOW_MIN = 3
RADIUS_WINDOW_REL = 0.20
EARLY_MARKER_MIN = 0.97  # 即試す条件のマーカー一致率 (シグネチャは完全一致が必須)
EARLY_ACCEPT = 0.5 + 0.5 * EARLY_MARKER_MIN  # 射影姿勢のスコア下限
MICRO_SCORE_PENALTY = 0.02  # 根拠の薄い micro 候補を他プロファイルの後ろに回す
FUNCTION_MATCH_MIN = 0.85  # 機能セル一致率の足切り
TRIPLE_TOL = 0.35  # 正三角形とみなす辺長のばらつき許容
EROSION_LEVELS = (0, 1, 2)  # 白マスクの収縮量 (画素)。細線を切るため複数試す
ENOUGH_ALIGNMENT_CANDIDATES = 10  # これだけ集まればアライメント探索を打ち切る
# ロケータ間距離 / (sqrt(3) * pitch) が取りうる範囲 = リング数 (R-1 または R-2)。
# ピッチ推定には数 % の系統誤差が乗るので、有効半径 (5..40) より広めに取る。
RING_RANGE = (2.5, 45.0)
PITCH_REL_TOL = 0.06  # ピッチ推定の相対誤差 (整数リング数判定の許容幅に使う)
MIN_RUN = 2  # ラベリング時に無視するラン長 (1 画素幅の細線・ノイズを捨てる)
ANNULUS_DARKNESS_MIN = 0.85  # ロケータ中心まわりが黒である割合の下限
OUTER_LIGHTNESS_MIN = 0.75  # 2 重リング外側が白である割合の下限
MARKER_MATCH_MIN = 0.90  # マーカーセルだけの一致率の足切り (射影歪み時の一次選抜)
ALIGNMENT_SEARCH_RADIUS = 3.0  # アライメント割り当ての許容距離 (セルピッチ単位)
PERSPECTIVE_BONUS = 0.03  # 6 点ホモグラフィ姿勢を優先する下駄
PERSPECTIVE_ITERS = 3  # アライメント割り当ての反復回数
SIGNATURE_MATCH_MIN = 0.99  # 射影姿勢を採用する際に要求するシグネチャ一致率
# 消失訂正の確信度しきい値。0 = 消失なし。順に緩めて再試行する。
ERASURE_LIMITS = (0.0, 0.35, 0.6)


# --- 二値化 -------------------------------------------------------------------


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    if np.count_nonzero(hist) < 2:  # 単色画像
        return int(gray.flat[0]) + 1
    total = gray.size
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.nanargmax(sigma_b))


@dataclass
class Binarized:
    """グレースケール画像と、画素ごとの閾値・黒マスク。

    thresh はスカラー (大域 Otsu) でも 2 次元配列 (適応的) でもよい。
    """

    gray: np.ndarray
    thresh: np.ndarray | float
    dark: np.ndarray
    name: str = "otsu"
    rgb: np.ndarray | None = None  # (H, W, 3) uint8。カラーパレット判定用

    def threshold_at(self, x: int, y: int) -> float:
        t = self.thresh
        return float(t) if np.isscalar(t) else float(t[y, x])


def binarize_otsu(gray: np.ndarray, rgb: np.ndarray | None = None) -> Binarized:
    t = otsu_threshold(gray)
    return Binarized(gray=gray, thresh=float(t), dark=gray < t, name="otsu",
                     rgb=rgb)


def local_mean_std(gray: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """半径 r の箱窓での局所平均・標準偏差 (積分画像で O(1)/画素)。

    画像を r 画素のゼロで囲ってから積分すると、端で切り詰めた窓の和が
    「はみ出し部分 = 0」として純粋なスライス演算だけで出せる
    (clip した添字での fancy indexing は 8 回の全画素 gather になり 2 倍以上遅い)。
    総和は int64 の cumsum なので桁落ちもない。
    """
    h, w = gray.shape
    a = np.pad(gray.astype(np.int64), r)
    i1 = np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))
    i2 = np.pad(np.cumsum(np.cumsum(a * a, 0), 1), ((1, 0), (1, 0)))
    d = 2 * r + 1

    def box(i: np.ndarray) -> np.ndarray:
        return (i[d:d + h, d:d + w] - i[:h, d:d + w]
                - i[d:d + h, :w] + i[:h, :w])

    ys, xs = np.arange(h), np.arange(w)
    ny = np.minimum(ys + r + 1, h) - np.maximum(ys - r, 0)
    nx = np.minimum(xs + r + 1, w) - np.maximum(xs - r, 0)
    count = ny[:, None] * nx[None, :]

    mean = box(i1) / count
    var = np.maximum(box(i2) / count - mean * mean, 0.0)
    return mean, np.sqrt(var)


def binarize_sauvola(gray: np.ndarray, radius: int, k: float = 0.2,
                     dynamic_range: float = 128.0,
                     rgb: np.ndarray | None = None) -> Binarized:
    """Sauvola 法の適応的二値化。照明ムラのある実写向け。

    T = mean * (1 + k * (std / R - 1))
    一様な明るい領域では std ~ 0 となり T ~ 0.8*mean まで下がるので、
    ノイズを黒と誤判定しにくい。
    """
    mean, std = local_mean_std(gray, radius)
    thresh = mean * (1.0 + k * (std / dynamic_range - 1.0))
    return Binarized(gray=gray, thresh=thresh, dark=gray < thresh,
                     name=f"sauvola(r={radius})", rgb=rgb)


# --- 連結成分 -----------------------------------------------------------------


@dataclass
class Component:
    area: int
    cx: float
    cy: float
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


def _extract_runs(mask: np.ndarray, min_run: int):
    """全行のランを一度に取り出す。戻り値は (行, 開始x, 終了x[排他])。

    行ごとに np.diff を呼ぶと numpy の呼び出しコストを行数ぶん払う
    (1400 行で 20ms 超)。2 次元のまま 1 回で済ませる。
    ランは行優先・行内は x 昇順に並ぶ (np.nonzero の走査順)。
    """
    h, w = mask.shape
    pad = np.zeros((h, w + 2), np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    ys, xs = np.nonzero(d == 1)
    xe = np.nonzero(d == -1)[1]
    if min_run > 1 and xs.size:
        keep = (xe - xs) >= min_run
        ys, xs, xe = ys[keep], xs[keep], xe[keep]
    return ys, xs, xe


def _run_adjacency(ys, xs, xe, h: int, w: int):
    """上下に隣接 (4 近傍) するランの組 (i, j) を返す。

    ランは行優先ソート済みなので、行 y のランと重なる行 y-1 のランは
    **連続した区間**になる。その両端を searchsorted で一括に求める
    (総当たりの二重ループを消す)。
    重なりの条件は `xs_i < xe_j かつ xs_j < xe_i`。
    """
    n = xs.size
    row_start = np.searchsorted(ys, np.arange(h + 1))
    key = w + 2  # 行をまたがない大きさ
    gstart = ys * key + xs
    gend = ys * key + xe
    prev = ys - 1
    lo = np.searchsorted(gend, prev * key + xs, side="right")
    hi = np.searchsorted(gstart, prev * key + xe, side="left") - 1
    # 直前の行の範囲に収める
    lo = np.maximum(lo, row_start[np.maximum(prev, 0)])
    hi = np.minimum(hi, row_start[ys] - 1)
    counts = np.where((ys > 0) & (lo <= hi), hi - lo + 1, 0)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    j = np.repeat(np.arange(n), counts)
    offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return lo[j] + offset, j


def label_components(mask: np.ndarray, min_run: int = MIN_RUN) -> list[Component]:
    """4 近傍のラン長ベース連結成分ラベリング。

    min_run 未満のランは捨てる。1 画素幅の線やノイズを落とすためで、
    探しているのはセル大 (数画素幅以上) の塊なので影響がなく、
    扱うラン数が大きく減って速くなる。

    ラン抽出・隣接判定・集計はすべて numpy で一括に行い、Python のループは
    union-find の統合だけに残す (組の数はラン数程度で、ここは元々安い)。
    """
    h, w = mask.shape
    ys, xs, xe = _extract_runs(mask, min_run)
    n = xs.size
    if n == 0:
        return []

    parent = np.arange(n, dtype=np.int64)

    ia, ib = _run_adjacency(ys, xs, xe, h, w)
    # 統合は Python の list で行う。ndarray の要素アクセスは 1 回ごとに
    # int 変換が入り、この密なループでは list の 3 倍近く遅い。
    par: list = parent.tolist()
    for a, b in zip(ia.tolist(), ib.tolist()):
        while par[a] != a:  # find(a) をインライン展開 (200 万回呼ばれる)
            par[a] = par[par[a]]
            a = par[a]
        while par[b] != b:
            par[b] = par[par[b]]
            b = par[b]
        if a != b:
            if a < b:
                par[b] = a
            else:
                par[a] = b
    parent = np.array(par, dtype=np.int64)

    # 経路圧縮: 親をたどり切るまで自分自身に置き換える (log 回で収束)
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    roots = parent

    # 根ごとの集計 (ソートしてから reduceat で一気に)
    order = np.argsort(roots, kind="stable")
    r = roots[order]
    starts = np.flatnonzero(np.concatenate(([True], r[1:] != r[:-1])))
    ys_o, xs_o, xe_o = ys[order], xs[order], xe[order]
    length = (xe_o - xs_o).astype(np.int64)
    area = np.add.reduceat(length, starts)
    sum_x = np.add.reduceat((xs_o + xe_o - 1) * length, starts) / 2.0
    sum_y = np.add.reduceat(ys_o * length, starts)
    x0 = np.minimum.reduceat(xs_o, starts)
    x1 = np.maximum.reduceat(xe_o - 1, starts)
    y0 = np.minimum.reduceat(ys_o, starts)
    y1 = np.maximum.reduceat(ys_o, starts)

    cx = sum_x / area
    cy = sum_y / area
    return [Component(int(a), float(px), float(py), int(a0), int(a1),
                      int(b0), int(b1))
            for a, px, py, a0, a1, b0, b1
            in zip(area.tolist(), cx.tolist(), cy.tolist(), x0.tolist(),
                   x1.tolist(), y0.tolist(), y1.tolist())]


# --- ロケータ候補 -------------------------------------------------------------


@dataclass
class LocatorCandidate:
    cx: float
    cy: float
    pitch: float
    darkness: float  # 半径 0.85*pitch の円周が黒である割合
    outer_lightness: float = 0.0  # 半径 1.85*pitch の円周が白である割合

    @property
    def is_double_ring(self) -> bool:
        """白中心 -> 黒リング -> 白リング の 2 重リングか (robust プロファイル)。"""
        return self.darkness >= ANNULUS_DARKNESS_MIN and self.outer_lightness >= OUTER_LIGHTNESS_MIN


def erode(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """8 近傍で k 回収縮する。1〜2 画素幅のヘアラインを切断するために使う。"""
    m = mask
    for _ in range(max(0, k)):
        p = np.pad(m, 1, constant_values=False)
        m = (p[0:-2, 0:-2] & p[0:-2, 1:-1] & p[0:-2, 2:] &
             p[1:-1, 0:-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
             p[2:, 0:-2] & p[2:, 1:-1] & p[2:, 2:])
    return m


def find_locator_candidates(dark: np.ndarray, min_area: int = 12) -> list[LocatorCandidate]:
    """「黒に囲まれた 1 セル大の白」= ロケータ中心の候補を集める。

    ここが検出全体の最大コスト (実測で約 7 割) だが、間引いたマスクで粗く探す
    案は「ぼかし / 回転で真のロケータを取りこぼすのに、それらしい三つ組は
    できてしまう」ため復号率が落ちた (1.3 倍速に対し退行 3 件)。等倍で探す。
    """
    return _find_blob_markers(dark, dark_center=False, min_area=min_area)


def find_alignment_candidates(dark: np.ndarray, min_area: int = 12
                              ) -> list[LocatorCandidate]:
    """「白に囲まれた 1 セル大の黒」= アライメント中心の候補を集める。

    ロケータとは白黒が反転しているだけなので、同じ処理を極性を変えて走らせる。
    """
    return _find_blob_markers(dark, dark_center=True, min_area=min_area)


def _find_blob_markers(dark: np.ndarray, dark_center: bool,
                       min_area: int = 12) -> list[LocatorCandidate]:
    """1 セル大の孤立領域で、周囲が反対色に囲まれているものを集める。

    描画では隣接する黒セルの間に白のセパレータが、白セルの周囲に黒の細線が入る
    ため、白も黒もセル頂点で細い線を介して繋がっている。そこでマスクを 0/1/2
    画素ぶん収縮させた 3 通りで連結成分を取り、細線を切断してからセル状の
    領域を拾う。収縮量はセルサイズで必要量が変わるので複数試して統合する。
    """
    h, w = dark.shape
    fill = dark if dark_center else ~dark
    cands: list[LocatorCandidate] = []
    kept: list[LocatorCandidate] = []
    for k in EROSION_LEVELS:
        light_k = erode(fill, k)
        dark_k = ~light_k  # 細線を埋めた反対色マスク
        passed: list[tuple[float, float, float]] = []  # (cx, cy, pitch)
        for c in label_components(light_k):
            if c.area < min_area:
                continue
            if c.x0 <= k or c.y0 <= k or c.x1 >= w - 1 - k or c.y1 >= h - 1 - k:
                continue  # 画像端に接する = 背景
            if c.width > w * 0.2 or c.height > h * 0.2:
                continue
            ratio = c.width / max(1, c.height)
            if not (0.6 < ratio < 1.7):
                continue
            if c.area / (c.width * c.height) < 0.5:
                continue
            # 収縮で痩せたぶんを戻すと、白セルの外接幅 ~= セルピッチ
            passed.append((c.cx, c.cy, (c.width + c.height) / 2.0 + 2 * k))
        if not passed:
            continue
        # 円周の黒率は候補ぶんまとめて測る (1 個ずつだと 1 フレーム 1.2 万回の
        # Python ループになっていた)
        arr = np.array(passed, dtype=np.float64)
        darkness = _annulus_darkness_batch(dark_k, arr[:, 0], arr[:, 1],
                                           arr[:, 2] * 0.85)
        ok = darkness >= ANNULUS_DARKNESS_MIN
        if not dark_center and ok.any():
            # ロケータのみ: さらに外側が白なら 2 重リング (robust プロファイル)
            outer = np.zeros(len(passed))
            outer[ok] = 1.0 - _annulus_darkness_batch(
                dark, arr[ok, 0], arr[ok, 1], arr[ok, 2] * 1.85)
        else:
            outer = np.zeros(len(passed))
        for i in np.flatnonzero(ok):
            cands.append(LocatorCandidate(arr[i, 0], arr[i, 1], arr[i, 2],
                                          float(darkness[i]), float(outer[i])))

        kept = _dedupe_candidates(cands, limit=80 if dark_center else 40)
        # ロケータは正三角形をなす組が既にあれば、これ以上収縮させても得はない。
        # アライメントは単独では見分けが付かないので数を確保したいが、
        # 十分に集まったら打ち切る (1 段が全画素ラベリング 1 回ぶん重い)。
        if dark_center:
            if len(kept) >= ENOUGH_ALIGNMENT_CANDIDATES:
                break
        elif _equilateral_triples(kept):
            break
    return kept


def _dedupe_candidates(cands: list[LocatorCandidate], limit: int = 40
                       ) -> list[LocatorCandidate]:
    """収縮量違いで重複して拾った同じセルをまとめる。"""
    kept: list[LocatorCandidate] = []
    for c in sorted(cands, key=lambda t: (-t.darkness, t.pitch)):
        if all(math.dist((c.cx, c.cy), (o.cx, o.cy)) > 0.5 * min(c.pitch, o.pitch)
               for o in kept):
            kept.append(c)
        if len(kept) >= limit:
            break
    return kept


_ANNULUS_ANGLES = np.stack([np.cos(np.linspace(0, 2 * math.pi, 48, endpoint=False)),
                            np.sin(np.linspace(0, 2 * math.pi, 48, endpoint=False))])


def _annulus_darkness_batch(dark: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                            radius: np.ndarray) -> np.ndarray:
    """各 (cx, cy) を中心とする半径 radius の円周が黒である割合。(M,) を返す。"""
    h, w = dark.shape
    x = np.rint(cx[:, None] + radius[:, None] * _ANNULUS_ANGLES[0][None, :])
    y = np.rint(cy[:, None] + radius[:, None] * _ANNULUS_ANGLES[1][None, :])
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    xi = np.clip(x, 0, w - 1).astype(np.int64)
    yi = np.clip(y, 0, h - 1).astype(np.int64)
    hit = dark[yi, xi] & inside
    total = inside.sum(axis=1)
    return np.divide(hit.sum(axis=1), total,
                     out=np.zeros(len(cx)), where=total > 0)


# --- アフィン変換とサンプリング -----------------------------------------------


def solve_affine(model: list[tuple[float, float]],
                 image: list[tuple[float, float]]) -> np.ndarray:
    """3 点対応から 3x3 の射影行列 (アフィン) を求める。"""
    M = np.array([[m[0], m[1], 1.0] for m in model], dtype=np.float64)
    P = np.array(image, dtype=np.float64)
    sol = np.linalg.solve(M, P)  # 3x2
    H = np.eye(3)
    H[:2, :] = sol.T
    return H


def solve_homography(model: list[tuple[float, float]],
                     image: list[tuple[float, float]]) -> np.ndarray:
    """4 点以上の対応から射影変換 (ホモグラフィ) を最小二乗で求める (DLT)。

    斜めから撮影した写真の台形歪みは、アフィン変換では表せずホモグラフィが要る。
    """
    n = len(model)
    if n < 4:
        raise ValueError("homography needs at least 4 correspondences")
    A = np.zeros((2 * n, 8), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    for i, ((mx, my), (px, py)) in enumerate(zip(model, image)):
        A[2 * i] = [mx, my, 1, 0, 0, 0, -mx * px, -my * px]
        A[2 * i + 1] = [0, 0, 0, mx, my, 1, -mx * py, -my * py]
        b[2 * i] = px
        b[2 * i + 1] = py
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


# 射影のユーティリティは pose.py に置き、姿勢オブジェクトと共有する



@lru_cache(maxsize=None)
def _match_table(radius: int, profile: str):
    """機能セルの同次モデル座標と期待値を numpy 配列で返す (キャッシュ)。

    姿勢候補は何百通りも評価するので、1 セルずつ射影すると効かない。
    マーカーとシグネチャを 1 本の配列にまとめ (前半 n_marker 個がマーカー)、
    1 回の射影で両方の一致率を出せるようにしておく。同次座標 (N,3) まで
    作っておくのは、呼び出しごとの hstack/ones の割り当てを消すため。
    """
    layout = get_layout(radius, profile)
    cells = (layout.locator_cells + layout.alignment_cells
             + layout.signature_cells)
    n_marker = len(layout.locator_cells) + len(layout.alignment_cells)
    xyh = np.ones((len(cells), 3), dtype=np.float64)
    xyh[:, :2] = [to_cartesian(c, 1.0) for c in cells]
    want = np.array([bool(layout.function_values[c]) for c in cells], dtype=bool)
    return xyh, want, n_marker


def _match_pair(bz: Binarized, H: np.ndarray, radius: int,
                profile: str) -> tuple[float, float]:
    """(マーカー一致率, シグネチャ一致率) を 1 回の射影で返す。範囲外は (-1,-1)。

    micro はシグネチャを持たないので、その場合のシグネチャ側は 1.0。
    """
    xyh, want, n_marker = _match_table(radius, profile)
    h, w = bz.gray.shape
    p = xyh @ H.T
    z = p[:, 2]
    if not np.all(np.isfinite(z)) or np.any(np.abs(z) < 1e-12):
        return -1.0, -1.0
    x = np.rint(p[:, 0] / z).astype(np.int64)
    y = np.rint(p[:, 1] / z).astype(np.int64)
    if x.min() < 0 or y.min() < 0 or x.max() >= w or y.max() >= h:
        return -1.0, -1.0
    hit = bz.dark[y, x] == want
    marker = float(np.count_nonzero(hit[:n_marker]) / n_marker)
    n_sig = len(want) - n_marker
    sig = float(np.count_nonzero(hit[n_marker:]) / n_sig) if n_sig else 1.0
    return marker, sig


def _marker_score(gray: np.ndarray, centers: np.ndarray, pitch: float,
                  dark_center: bool) -> np.ndarray:
    """候補中心ごとに「中心とリングの明暗差」を返す。大きいほどマーカーらしい。"""
    h, w = gray.shape
    inner = _UNIT_SAMPLES[:7] * (pitch * 0.6)
    ang = np.radians(np.arange(12) * 30.0)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * pitch

    def mean_at(offsets: np.ndarray) -> np.ndarray:
        xs = np.clip(np.rint(centers[:, 0:1] + offsets[None, :, 0]), 0, w - 1).astype(np.int64)
        ys = np.clip(np.rint(centers[:, 1:2] + offsets[None, :, 1]), 0, h - 1).astype(np.int64)
        return gray[ys, xs].mean(axis=1)

    c = mean_at(inner)
    r = mean_at(ring)
    return (r - c) if dark_center else (c - r)


def _centroid_of(mask: np.ndarray, x: float, y: float,
                 pitch: float, rounds: int = 2) -> tuple[float, float]:
    """(x, y) 付近にある「マスクが立っている領域」の重心を返す。"""
    h, w = mask.shape
    r = max(2, int(round(pitch * 0.45)))
    for _ in range(rounds):
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return x, y
        ys, xs = np.mgrid[y0:y1, x0:x1]
        sel = mask[y0:y1, x0:x1] & (((xs - x) ** 2 + (ys - y) ** 2) <= r * r)
        if int(sel.sum()) < 4:
            return x, y
        x = float(xs[sel].mean())
        y = float(ys[sel].mean())
    return x, y


def _centroid_refine(bz: Binarized, x: float, y: float, pitch: float,
                     dark_center: bool) -> tuple[float, float]:
    """中心セルの塗り領域の重心を取り、サブピクセル精度に引き上げる。

    格子探索だけでは 1 セル近くずれることがあるので、見つけた極大のまわりで
    「期待する色の画素」の重心を求め直す。
    """
    return _centroid_of(bz.dark if dark_center else ~bz.dark, x, y, pitch)


def refine_marker(bz: Binarized, cx: float, cy: float, pitch: float,
                  dark_center: bool, span: float = 1.5) -> tuple[float, float]:
    """予測位置のまわりでマーカー中心を探し直す (格子探索 -> 重心)。

    アフィン近似で予測した位置は射影歪みのぶんズレるので、実際の模様に合わせて
    引き込む。QR のアライメントパターン探索と同じ考え方。
    """
    gray = bz.gray
    x, y = cx, cy
    for reach, step in ((span * pitch, pitch / 5.0), (pitch / 3.0, pitch / 15.0)):
        n = max(1, int(round(reach / step)))
        d = np.arange(-n, n + 1) * step
        gx, gy = np.meshgrid(x + d, y + d)
        centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
        scores = _marker_score(gray, centers, pitch, dark_center)
        best = int(np.argmax(scores))
        x, y = float(centers[best, 0]), float(centers[best, 1])
    return _centroid_refine(bz, x, y, pitch, dark_center)


REFINE_SPANS = (1.8, 1.0, 0.6)


def refine_homography(bz: Binarized, H: np.ndarray, radius: int, profile: str
                      ) -> np.ndarray | None:
    """アフィン姿勢からアライメントマーカーを探し直し、6 点でホモグラフィを解く。

    1 回では引き込めないほど初期姿勢がずれていることがあるので、探索範囲を
    狭めながら数回繰り返す。1 個でも当たればホモグラフィが改善し、
    次の周回で残りの予測位置も正確になる。
    """
    layout = get_layout(radius, profile)
    if not layout.alignment_centers:
        return None
    h, w = bz.gray.shape

    # ロケータ 3 点はそのまま (検出済みの実測値に一致している)
    model_loc = [to_cartesian(c, 1.0) for c in layout.locator_centers]
    image_loc = [project(H, c) for c in layout.locator_centers]

    cur = H
    improved = False
    for span in REFINE_SPANS:
        model_pts = list(model_loc)
        image_pts = list(image_loc)
        for cell in layout.alignment_centers:
            px, py = project(cur, cell)
            if not (0 <= px < w and 0 <= py < h):
                return cur if improved else None
            # 局所的なセルピッチ (射影歪みで場所ごとに変わる)
            nx, ny = project(cur, (cell[0] + 1, cell[1]))
            pitch = math.hypot(nx - px, ny - py)
            if not (3.0 <= pitch <= max(w, h)):
                return cur if improved else None
            model_pts.append(to_cartesian(cell, 1.0))
            image_pts.append(refine_marker(bz, px, py, pitch, True, span=span))
        try:
            cur = solve_homography(model_pts, image_pts)
        except (ValueError, np.linalg.LinAlgError):
            return cur if improved else None
        improved = True
    return cur if improved else None


@dataclass
class SampledGrid:
    bits: dict[Axial, int]
    confidence: dict[Axial, float]  # 0..1。閾値からの距離を正規化したもの
    # RGB は遅延評価。カラーパレットのときしか要らないのに、姿勢候補ごとに
    # 毎回計算すると無駄が大きい (実測で全体の 4% がここだった)。
    _rgb_src: tuple | None = None
    _rgb: dict[Axial, tuple[float, float, float]] | None = None

    @property
    def rgb(self):
        if self._rgb is None and self._rgb_src is not None:
            self._rgb = _sample_rgb(*self._rgb_src)
        return self._rgb

    def weak_cells(self, limit: float) -> list[Axial]:
        return [c for c, v in self.confidence.items() if v < limit]

    def with_palette(self, layout, palette: str) -> "SampledGrid":
        """データセルをパレット色で読み直した格子を返す。

        機能セル (マーカー / シグネチャ / フォーマット) は常に黒白なので
        モノクロ判定のまま据え置く。
        """
        if palette == DEFAULT_PALETTE:
            return self
        rgb = self.rgb
        if rgb is None:
            return self
        values = dict(self.bits)
        conf = dict(self.confidence)
        cells = layout.data_cells
        arr = np.array([rgb[c] for c in cells], dtype=np.float64)
        vs, cs = classify_batch(palette, arr)
        values.update(zip(cells, vs.tolist()))
        conf.update(zip(cells, cs.tolist()))
        return SampledGrid(bits=values, confidence=conf, _rgb=rgb)


# セル中心まわりの相対サンプル点 (中心 + 半径 0.5 / 0.85 の六方向)
_UNIT_SAMPLES = np.array(
    [(0.0, 0.0)]
    + [(0.50 * math.cos(math.radians(60 * i)), 0.50 * math.sin(math.radians(60 * i)))
       for i in range(6)]
    + [(0.85 * math.cos(math.radians(60 * i + 30)),
        0.85 * math.sin(math.radians(60 * i + 30))) for i in range(6)],
    dtype=np.float64,
)


@lru_cache(maxsize=None)
def _cell_models(radius: int, profile: str):
    """全セルのモデル座標と、隣接セルのモデル座標。

    サンプリングのたびに作り直すと `to_cartesian` が 1 フレーム 480 万回に達する。
    """
    cells = get_layout(radius, profile).cells
    model = np.array([to_cartesian(c, 1.0) for c in cells], dtype=np.float64)
    nbr = np.array([to_cartesian((c[0] + 1, c[1]), 1.0) for c in cells],
                   dtype=np.float64)
    return model, nbr


def sample_grid(bz: Binarized, H: np.ndarray, radius: int,
                profile: str = DEFAULT_PROFILE,
                sample_radius_ratio: float = 0.30) -> SampledGrid | None:
    """全セル中心をサンプリングし、ビットと確信度を返す。

    セルごとに隣接セルへの投影距離から標本半径を決めるので、射影変換で
    手前と奥のセルサイズが違っていても追従する。

    確信度は「セル平均輝度と閾値の差」を同じビットのセルの中央値で正規化した値。
    汚れ・遮蔽・ボケで白黒がはっきりしないセルはここが低くなり、
    RS の消失位置 (erasure) として扱える。
    """
    layout = get_layout(radius, profile)
    gray = bz.gray
    h, w = gray.shape
    cells = layout.cells

    model, nbr = _cell_models(radius, profile)
    pts = project_xy(H, model)
    pts_n = project_xy(H, nbr)
    rad = np.hypot(*(pts_n - pts).T) * sample_radius_ratio  # (N,)
    if not np.all(np.isfinite(pts)) or np.any(rad <= 0):
        return None

    xs = np.rint(pts[:, 0]).astype(np.int64)
    ys = np.rint(pts[:, 1]).astype(np.int64)
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= w or ys.max() >= h:
        return None

    # (N, M) の標本点
    sx = np.clip(np.rint(pts[:, 0:1] + rad[:, None] * _UNIT_SAMPLES[None, :, 0]),
                 0, w - 1).astype(np.int64)
    sy = np.clip(np.rint(pts[:, 1:2] + rad[:, None] * _UNIT_SAMPLES[None, :, 1]),
                 0, h - 1).astype(np.int64)
    means = gray[sy, sx].mean(axis=1)

    thr = bz.thresh
    t = np.full(len(cells), float(thr)) if np.isscalar(thr) else thr[ys, xs]
    values = (means < t).astype(np.int64)
    margins = np.abs(means - t)

    # 確信度: 同じビットのセルの余裕の中央値で正規化する
    conf = np.ones(len(cells))
    for bit in (0, 1):
        sel = values == bit
        if not sel.any():
            continue
        ref = float(np.median(margins[sel]))
        if ref > 0:
            conf[sel] = np.minimum(1.0, margins[sel] / ref)

    src = (bz, layout, sx, sy, values, cells) if bz.rgb is not None else None
    return SampledGrid(bits={c: int(v) for c, v in zip(cells, values)},
                       confidence={c: float(v) for c, v in zip(cells, conf)},
                       _rgb_src=src)


def _sample_rgb(bz: Binarized, layout, sx: np.ndarray, sy: np.ndarray,
                mono: np.ndarray, cells) -> dict[Axial, tuple[float, float, float]]:
    """セルごとの平均 RGB を、マーカーの黒白を基準に正規化して返す。

    マーカーは必ず黒か白なので、シンボル内から照明・ホワイトバランスの基準を
    取り出せる。これでチャンネルごとに [0, 1] へ写してからパレット判定する。
    """
    rgb = bz.rgb
    means = rgb[sy, sx].mean(axis=1)  # (N, 3)
    index = {c: i for i, c in enumerate(cells)}

    dark_idx = [index[c] for c, v in layout.function_values.items() if v]
    light_idx = [index[c] for c, v in layout.function_values.items() if not v]
    if len(dark_idx) >= 3 and len(light_idx) >= 3:
        black = np.median(means[dark_idx], axis=0)
        white = np.median(means[light_idx], axis=0)
    else:  # マーカーが取れないときは全体の分位点で代用
        black = np.percentile(means, 5, axis=0)
        white = np.percentile(means, 95, axis=0)
    span = np.maximum(white - black, 1.0)
    norm = np.clip((means - black) / span, -0.3, 1.3)
    return {c: (float(norm[i, 0]), float(norm[i, 1]), float(norm[i, 2]))
            for c, i in index.items()}


# --- 復号 ---------------------------------------------------------------------


@dataclass
class DetectionReport:
    candidates: int = 0
    triples: int = 0
    attempts: int = 0
    binarization: str = ""
    erasures: int = 0
    alignments: int = 0
    profile: str = ""
    palette: str = ""
    work_size: int = 0  # 成功したときの作業画像の長辺
    binarizer: str = ""  # 成功したときの二値化の種類キー
    # 検出したファインダ候補の座標 (入力画像のスケール)。
    # 復号に失敗しても「マーカーは見えている」ことを可視化できるようにする。
    finder_candidates: list[tuple[float, float]] = field(default_factory=list)
    perspective: bool = False
    errors: list[str] = field(default_factory=list)


def _load_gray(source) -> tuple[np.ndarray, float]:
    gray, _, scale = _load_image(source)
    return gray, scale


def _load_image(source, max_size: int = MAX_WORK_SIZE
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """(グレースケール, RGB, 縮小率) を返す。"""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = img.convert("RGB")
    scale = 1.0
    m = max(img.size)
    if m > max_size:
        scale = max_size / m
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.LANCZOS)
    rgb = np.asarray(img, dtype=np.uint8)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return gray, rgb, scale


def work_sizes(long_side: int) -> list[int]:
    """試す作業画像の長辺を大きい順に返す。

    実写 (画面や書類を撮影) では、縮小すると UI の細かい白領域やモアレが
    平滑化されてロケータ候補のノイズが激減し、**検出率が上がる**。
    実写映像では 1400px で 24%、600px で 66%、420px で 91% だった。
    一方で小さくしすぎるとセルが潰れるので、小さい方から順に試して
    駄目なら大きくしていく (小さい方が安いので、当たれば速い)。
    """
    first = min(long_side, MAX_WORK_SIZE)
    out = [first] + [t for t in WORK_SIZE_LADDER if t <= first * 0.75]
    return sorted(set(out))


BINARIZERS = ("otsu", "sauvola8", "sauvola16")
# 二値化 1 回のおおよその相対コスト (Sauvola は積分画像を 2 枚作る)
BINARIZER_COST = {"otsu": 1.0, "sauvola8": 3.0, "sauvola16": 3.0}


def sweep_order(sizes, kinds=BINARIZERS,
                allow_inverted: bool = False) -> list[tuple[str, int, bool]]:
    """(二値化, 作業サイズ, 反転フラグ) を **安い順** に並べる。

    コストはおおむね「画素数 x 二値化の重み」。種類を外側にすると
    otsu@1400 (高い) を sauvola@420 (安い) より先に払うことになる。
    どの組み合わせで当たるかは画像次第 (実測で otsu 75 / sauvola 30) なので、
    全部試すのは変えずに、安い方から並べて当たるまでの時間を縮める。

    allow_inverted=True のときは、通常向きを一巡したあとに白黒反転版を試す。
    正常な画像は通常向きで先に当たって打ち切られるので、探索が増えるのは
    「未検出のとき」だけ。
    """
    pairs = sorted(((k, s) for k in kinds for s in sizes),
                   key=lambda ks: ks[1] * ks[1] * BINARIZER_COST[ks[0]])
    order = [(k, s, False) for k, s in pairs]
    if allow_inverted:
        order += [(k, s, True) for k, s in pairs]
    return order


def make_binarized(kind: str, gray: np.ndarray, rgb: np.ndarray | None,
                   inverted: bool = False) -> Binarized:
    """種類を指定して二値化する。Sauvola は 1 回 50ms 前後かかる。

    inverted=True は白黒反転したコード (明暗が逆) を読むための版。
    グレースケール (と RGB) を反転してから二値化するので、以降の
    検出・サンプリングはすべて通常の向きとして処理できる。
    """
    if inverted:
        gray = (255 - gray).astype(np.uint8)
        rgb = None if rgb is None else (255 - rgb).astype(np.uint8)
    if kind == "otsu":
        bz = binarize_otsu(gray, rgb=rgb)
    else:
        frac = 8 if kind == "sauvola8" else 16
        bz = binarize_sauvola(gray, max(8, min(gray.shape) // frac), rgb=rgb)
    if inverted:
        bz.name += " inv"
    return bz


def _binarizations(gray: np.ndarray, rgb: np.ndarray | None = None):
    """試す二値化を順に返す (逐次生成なので、通ったら以降は作らない)。"""
    for kind in BINARIZERS:
        yield make_binarized(kind, gray, rgb)


def decode_image(source, radius_hint: int | None = None,
                 report: DetectionReport | None = None,
                 profile: str | None = None,
                 exclude: list[tuple[float, float, float]] | None = None,
                 binarizers: tuple[str, ...] | None = None,
                 allow_inverted: bool = True) -> DecodeResult:
    """PNG / JPEG などの画像からシンボルを読み取る。

    profile を省略するとすべてのプロファイルを試す。
    exclude に (x, y, r) を渡すと、その円内のロケータ候補を無視する
    (複数シンボルを順に読むときに、既に読んだものを除外するため)。
    allow_inverted=True (既定) なら白黒反転したコードも読む
    (未検出のときだけ探索が増える)。allow_inverted=False で無効化。
    """
    img = source if isinstance(source, Image.Image) else Image.open(source)
    rep = report if report is not None else DetectionReport()
    profiles = [profile] if profile else list(PROFILES)
    sizes = work_sizes(max(img.size))
    cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    kinds = binarizers or BINARIZERS
    res = _decode_sizes(img, sizes, profiles, radius_hint, rep, exclude, cache,
                        kinds, allow_inverted)

    # 粗いサイズで当たった場合は、判明した半径 / プロファイルを手がかりに
    # 最大サイズで引き直す。座標の量子化が細かくなり、ジオメトリの精度が戻る。
    if res.geometry is not None and rep.work_size != sizes[-1]:
        fine = DetectionReport()
        try:
            better = _decode_sizes(img, [sizes[-1]], [res.profile], res.radius,
                                   fine, exclude, cache, kinds, allow_inverted)
        except MutsumeError:
            better = None
        if better is not None and better.payload == res.payload:
            res.geometry = better.geometry
            rep.work_size = fine.work_size
    return res


def _decode_sizes(img, sizes: list[int], profiles: list[str],
                  radius_hint: int | None, rep: DetectionReport,
                  exclude, cache: dict,
                  kinds: tuple[str, ...] = BINARIZERS,
                  allow_inverted: bool = False) -> DecodeResult:
    """作業サイズ x 二値化の総当たり。

    ループの順序が効く。二値化の種類を外側にして **まず安価な Otsu で全サイズを
    一巡**する。Sauvola は 1 回 50ms 前後かかるので、サイズを外側にすると
    各サイズで毎回それを払う (実測で 1 フレーム 6.2 秒 -> 1.1 秒 の差)。
    サイズは小さい方から見る。実写では縮小した方が当たりやすく、かつ安い
    (同じ検出率で 5 倍速)。
    """
    failures: list[str] = []
    for kind, size, inv in sweep_order(sizes, kinds, allow_inverted):
        if size not in cache:
            cache[size] = _load_image(img, size)
        gray, rgb, scale = cache[size]
        bz = make_binarized(kind, gray, rgb, inv)
        rep.work_size, rep.binarization, rep.binarizer = size, bz.name, kind
        try:
            return _decode_binarized(bz, radius_hint, rep, profiles, scale,
                                     exclude)
        except MutsumeError as e:
            failures.append(f"{size}px/{bz.name}: {e}")
    raise MutsumeError(" | ".join(failures[-3:]))


def _exclusion_of(res: DecodeResult) -> tuple[float, float, float] | None:
    g = res.geometry
    if g is None:
        return None
    x0, y0, x1, y1 = g.bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2,
            0.5 * max(x1 - x0, y1 - y0) * EXCLUDE_MARGIN)


def _track_one(img, g, rep: DetectionReport, binarize,
               exclude: list[tuple[float, float, float]]) -> DecodeResult | None:
    """前フレームの姿勢 (Geometry) を初期値に、検出を跳ばして復号だけ試す。

    ファインダ 3 点を前回の位置の近くで取り直してアフィンを解き直すので、
    フレーム間の小さな動き (手ブレ程度) には追従する。当たれば
    二値化 + サンプリング + 復号だけで済み、ロケータ検出 (ラベリング)・
    三つ組列挙・姿勢総当たりを丸ごと省ける。

    binarize は (kind, size) -> (Binarized, scale) の共有キャッシュ。
    複数ヒントや後続の全探索と二値化を共有する (Sauvola は 1 回数十 ms)。
    """
    size = min(max(img.size), MAX_WORK_SIZE)

    # 前回の中心が除外領域 (このフレームで既に読んだ別シンボル) なら重複
    if any(math.dist(g.center, (ex, ey)) < er for ex, ey, er in exclude):
        return None

    layout = get_layout(g.radius, g.profile)
    for kind in TRACK_BINARIZERS:
        bz, scale = binarize(kind, size)
        H = np.diag([scale, scale, 1.0]) @ g.homography
        pitch = g.cell_size * scale
        gray = bz.gray
        h, w = gray.shape
        model: list[tuple[float, float]] = []
        pts: list[tuple[float, float]] = []
        ok = True
        for cell in layout.locator_centers:
            px, py = project(H, cell)
            if not (0 <= px < w and 0 <= py < h):
                ok = False
                break
            model.append(to_cartesian(cell, 1.0))
            pts.append(refine_marker(bz, px, py, pitch, dark_center=False,
                                     span=TRACK_SPAN))
        if not ok:
            return None
        try:
            H2 = solve_affine(model, pts)
        except np.linalg.LinAlgError:
            continue
        marker, _sig = _match_pair(bz, H2, g.radius, g.profile)
        if marker < MARKER_MATCH_MIN:
            continue
        rep.work_size, rep.binarization, rep.binarizer = size, bz.name, kind
        pose = _Pose(1.0, g.radius, g.profile, H2)
        res = _try_pose(bz, pose, rep, scale)
        if res is not None:
            return res
    return None


def decode_image_all(source, max_symbols: int = 8, radius_hint: int | None = None,
                     profile: str | None = None,
                     report: DetectionReport | None = None,
                     hints: list | None = None,
                     hints_only: bool = False,
                     allow_inverted: bool = True) -> list[DecodeResult]:
    """画像に写っているコードを **すべて** 読み取る。

    1 つ読めたらその外形を除外領域に加えて探し直す、を繰り返す。
    同じシンボルを二度返さないよう、除外はロケータ候補の段階で効かせる。

    2 つめ以降も **条件 (作業サイズ・二値化) を絞らずに探索し直す**。
    1 枚に写った複数のコードは大きさも見え方も違い、当たる条件がそれぞれ違うので、
    1 つめの条件を流用すると大きく取りこぼす
    (実測: 全段 42% / サイズ固定 5% / 二値化固定 9%)。

    そのかわり **二値化とロケータ候補の抽出は (種類, サイズ) ごとに 1 回だけ**
    行ってキャッシュする。ここが検出コストの大半なので、
    2 個目以降の探索と「もう無い」の確認がほぼ姿勢探索だけになる。

    動画では hints に前フレームの `DecodeResult.geometry` を渡すと、
    検出を跳ばしてその姿勢の近傍だけで復号を試す (トラッキング)。
    hints_only=True なら、全ヒントが追従できた場合に全探索を省く。
    新しいシンボルはヒントでは見つからないので、呼び出し側で定期的に
    hints_only=False のフレームを挟むこと。

    allow_inverted=True (既定) なら白黒反転したコードも読む
    (未検出のときだけ探索が増える)。allow_inverted=False で無効化。
    """
    img = source if isinstance(source, Image.Image) else Image.open(source)
    rep = report if report is not None else DetectionReport()
    profiles = [profile] if profile else list(PROFILES)
    sizes = work_sizes(max(img.size))
    loaded: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    bins: dict[tuple[str, int, bool], tuple[Binarized, float]] = {}
    ctx: dict[tuple[str, int, bool], list] = {}

    # 二値化はトラッキングと全探索の両方で使うので (種類, サイズ) で共有する。
    # 別々に持つと、複数ヒント x 全探索フレームで同じ Sauvola (数十 ms) を
    # 何度も払うことになる。
    def binarize(kind: str, size: int, inv: bool = False) -> tuple[Binarized, float]:
        key = (kind, size, inv)
        if key not in bins:
            if size not in loaded:
                loaded[size] = _load_image(img, size)
            gray, rgb, scale = loaded[size]
            bins[key] = (make_binarized(kind, gray, rgb, inv), scale)
        return bins[key]

    def context(kind: str, size: int, inv: bool):
        key = (kind, size, inv)
        bz, scale = binarize(kind, size, inv)
        if key not in ctx:
            ctx[key] = find_locator_candidates(bz.dark)
        return bz, scale, ctx[key]

    out: list[DecodeResult] = []
    exclude: list[tuple[float, float, float]] = []

    tracked_all = True
    for g in (hints or []):
        if g is None or len(out) >= max_symbols:
            tracked_all = False
            continue
        res = _track_one(img, g, rep, binarize, exclude)
        if res is None:
            tracked_all = False
            continue
        out.append(res)
        ex = _exclusion_of(res)
        if ex is not None:
            exclude.append(ex)
    if hints_only and hints and tracked_all:
        return out
    order = sweep_order(sizes, allow_inverted=allow_inverted)
    hit_candidates: list[tuple[float, float]] = []
    for k in range(max_symbols - len(out)):
        found = None
        # 2 つめ以降は探索量を絞る。もう 1 つ写っているなら真の三つ組は上位に
        # 来るので、通常は取りこぼさない。一方「もう無い」を確かめる一巡は
        # 雑然とした画像だと非常に高くつく (実測でフレーム時間の 8 割)。
        budget = None if (k == 0 and not out) else EXTRA_SYMBOL_BUDGET
        for kind, size, inv in order:
            bz, scale, cands = context(kind, size, inv)
            rep.work_size, rep.binarization, rep.binarizer = size, bz.name, kind
            try:
                found = _decode_binarized(bz, radius_hint, rep, profiles,
                                          scale, exclude, cands, budget)
                break
            except MutsumeError:
                continue
        if found is None:
            break
        # 当たった条件でのファインダ候補を控える (呼び出し側の可視化用)。
        # 後続の失敗した一巡で上書きされないようにする。
        hit_candidates = list(rep.finder_candidates)
        out.append(found)
        ex = _exclusion_of(found)
        if ex is None:
            break
        exclude.append(ex)
    if hit_candidates:
        rep.finder_candidates = hit_candidates
    return out


def locate_image(source, radius_hint: int | None = None,
                 profile: str | None = None) -> Geometry:
    """復号せずに位置だけ欲しい場合の入口。Geometry を返す。

    実装としては通常の復号を行う (CRC が通って初めて姿勢が確定するため)。
    """
    res = decode_image(source, radius_hint=radius_hint, profile=profile)
    if res.geometry is None:  # pragma: no cover - 復号成功時は必ず入る
        raise MutsumeError("ジオメトリを取得できません")
    return res.geometry


@dataclass
class _Pose:
    score: float
    radius: int
    profile: str
    H: np.ndarray
    perspective: bool = False
    perfect: bool = False  # マーカーもシグネチャも完全一致 (即試す価値がある)


def _match_alignments(H: np.ndarray, layout, alignments: list[LocatorCandidate],
                      pitch: float, top: int = 2) -> list[list[tuple[float, float]]]:
    """予測位置の近くにある検出済みアライメントの組み合わせを列挙する。

    アライメント (黒 1 セル + 白リング) は単独では見分けが付かず、データ部にも
    同じ形が現れる。そこで各予測位置について近い順に数個ずつ拾い、組み合わせを
    ホモグラフィ + 機能セル一致率で評価して正解を選ぶ。
    """
    per_cell: list[list[tuple[float, float]]] = []
    for cell in layout.alignment_centers:
        px, py = project(H, cell)
        near = sorted(
            ((math.dist((px, py), (a.cx, a.cy)), (a.cx, a.cy)) for a in alignments),
            key=lambda t: t[0])
        picked = [p for d, p in near[:top] if d <= ALIGNMENT_SEARCH_RADIUS * pitch]
        if not picked:
            return []
        per_cell.append(picked)
    return [list(combo) for combo in itertools.product(*per_cell)]


def _nearest_alignments(H: np.ndarray, layout, alignments: list[LocatorCandidate]
                        ) -> list[tuple[float, float]]:
    out = []
    for cell in layout.alignment_centers:
        px, py = project(H, cell)
        a = min(alignments, key=lambda c: math.dist((px, py), (c.cx, c.cy)))
        out.append((a.cx, a.cy))
    return out


def _fit_perspective(bz: Binarized, H: np.ndarray, layout,
                     alignments: list[LocatorCandidate], pitch: float,
                     model: list[tuple[float, float]],
                     loc_img: list[tuple[float, float]],
                     radius: int, profile: str) -> tuple[float, np.ndarray] | None:
    """アライメントを割り当てて 6 点ホモグラフィを解く。最良の (score, H) を返す。

    初期割り当てはアフィン予測の近傍なので外すことがある。ホモグラフィを解いて
    予測し直す、を数回繰り返すと正しいマーカーに引き込まれる。
    """
    model6 = model + [to_cartesian(c, 1.0) for c in layout.alignment_centers]
    best: tuple[float, np.ndarray] | None = None
    for combo in _match_alignments(H, layout, alignments, pitch):
        cur = combo
        H6 = None
        for _ in range(PERSPECTIVE_ITERS):
            try:
                H6 = solve_homography(model6, loc_img + cur)
            except (ValueError, np.linalg.LinAlgError):
                H6 = None
                break
            nxt = _nearest_alignments(H6, layout, alignments)
            if nxt == cur:
                break
            cur = nxt
        if H6 is None:
            continue
        # シグネチャは半径と向きの両方に効く。ここを厳しくしないと
        # 「回転した姿勢が偶然通る」ケースが混ざる。
        marker, sig = _match_pair(bz, H6, radius, profile)
        if sig < SIGNATURE_MATCH_MIN:
            continue
        s = 0.5 * marker + 0.5 * sig
        if s >= FUNCTION_MATCH_MIN and (best is None or s > best[0]):
            best = (s, H6)
    return best


def _pose_candidates(bz: Binarized, triples, radius_hint: int | None, profile: str,
                     alignments: list[LocatorCandidate],
                     max_triples: int = MAX_TRIPLES):
    """ロケータ 3 点の対応付けと半径を順に試し、姿勢候補を **逐次** 生成する。

    ジェネレータにしてあるのは、「マーカーもシグネチャも完全一致した姿勢」が
    出た時点で呼び出し側が復号を試せるようにするため。当たれば残りの半径 x 対応
    (最大 200 通り超) を評価せずに済む。外れたら生成を続けるので、
    最悪ケースは総当たりのままで堅牢性は落ちない。

    robust プロファイルでは、アフィン姿勢でアライメントを割り当てたうえで
    6 点からホモグラフィを解き直す。これで射影歪み (斜めからの撮影) に対応する。
    """
    for pts, side in triples[:max_triples]:
        pitch = float(np.median([p.pitch for p in pts]))
        if pitch <= 0:
            continue
        # ロケータ中心はリング (R-1) または (R-2) のコーナー 0/2/4。相互距離は
        #   3 * ring * size = sqrt(3) * ring * pitch   (pitch = sqrt(3) * size)
        offset = 2 if profile == "robust" else 1
        r_est = int(round(side / (CELL_PITCH * pitch))) + offset

        lo, hi = min_radius(profile), max_radius(profile)
        # micro は半径 5..8 しかない。幾何推定が大きく外れていれば丸ごと飛ばす
        # (毎回 4 半径 x 6 対応を試すと、大きいシンボルの復号が遅くなる)
        if profile == "micro" and not (lo - 2 <= r_est <= hi + 2):
            continue
        if radius_hint:
            if not (lo <= radius_hint <= hi):
                continue  # このプロファイルには存在しない半径
            radii = [radius_hint]
        else:
            # 幾何推定の近傍だけ見る。三つ組が正しければ r_est はよく当たり、
            # 正しくなければ何を試しても通らない。全 34 通り試すのは、
            # 雑然とした実写で偽の三つ組が大量に出たときに効いてくる
            # (姿勢評価が 1 フレーム 3 万回まで膨らんでいた)。
            span = max(RADIUS_WINDOW_MIN, round(r_est * RADIUS_WINDOW_REL))
            radii = [r for r in range(r_est - span, r_est + span + 1)
                     if lo <= r <= hi]
            if not radii:
                radii = list(range(lo, hi + 1))
        # 幾何推定に近い半径から見る。シグネチャが完全一致した時点で打ち切れるので、
        # 推定が当たっていれば数評価で決まる (総当たりは外れたときの保険)
        radii.sort(key=lambda r: abs(r - r_est))
        can_stop = bool(get_layout(radii[0], profile).signature_cells) if radii else False

        image_pts = [(p.cx, p.cy) for p in pts]
        for radius in radii:
            layout = get_layout(radius, profile)
            model = [to_cartesian(c, 1.0) for c in layout.locator_centers]
            for perm in itertools.permutations(range(3)):
                try:
                    H = solve_affine(model, [image_pts[i] for i in perm])
                except np.linalg.LinAlgError:
                    continue
                bonus = 0.02 if radius == r_est else 0.0

                if layout.alignment_centers and len(alignments) >= 3:
                    best6 = _fit_perspective(bz, H, layout, alignments, pitch,
                                             model, [image_pts[i] for i in perm],
                                             radius, profile)
                    if best6 is not None:
                        # _fit_perspective はシグネチャ完全一致を要求済みなので、
                        # ここでの下限はマーカー側の許容 (= 0.5 + 0.5*marker)
                        yield _Pose(best6[0] + bonus + PERSPECTIVE_BONUS,
                                    radius, profile, best6[1], True,
                                    perfect=can_stop and best6[0] >= EARLY_ACCEPT)

                marker, sig = _match_pair(bz, H, radius, profile)
                if marker < 0:
                    continue
                full = 0.5 * marker + 0.5 * sig
                # 射影歪みがあるとシグネチャは外れるので、マーカー一致だけでも拾う
                if full < FUNCTION_MATCH_MIN and marker < MARKER_MATCH_MIN:
                    continue
                if profile == "micro":
                    # micro: シグネチャがなく順位付けの根拠はマーカーだけ。
                    # 根拠を持つ他プロファイルに競り負けるよう、わずかに割り引く
                    score = marker - MICRO_SCORE_PENALTY
                else:
                    # マーカーは半径が 1 ずれてもだいたい当たってしまうので、
                    # 半径と向きの両方に効くシグネチャを重く見る
                    score = 0.35 * marker + 0.65 * sig
                # 「即試す」条件: シグネチャは完全一致 (半径と向きを決める要)、
                # マーカーはノイズで数セル外れうるので少し許容する
                yield _Pose(score + bonus, radius, profile, H, False,
                            perfect=(can_stop and sig >= 1.0
                                     and marker >= EARLY_MARKER_MIN))


def _decode_binarized(bz: Binarized, radius_hint: int | None,
                      rep: DetectionReport, profiles: list[str],
                      scale: float = 1.0,
                      exclude: list[tuple[float, float, float]] | None = None,
                      cands: list[LocatorCandidate] | None = None,
                      budget: tuple[int, int] | None = None) -> DecodeResult:
    if cands is None:
        cands = find_locator_candidates(bz.dark)
    inv = 1.0 / scale if scale else 1.0
    if exclude:  # 既に読んだシンボルの領域を落とす (元画像座標で判定)
        cands = [c for c in cands
                 if all(math.dist((c.cx * inv, c.cy * inv), (ex, ey)) > er
                        for ex, ey, er in exclude)]
    rep.candidates = len(cands)
    rep.finder_candidates = [(c.cx * inv, c.cy * inv) for c in cands]
    if len(cands) < 3:
        raise MutsumeError(f"ロケータ候補が見つかりません ({len(cands)} 個)")

    poses: list[_Pose] = []
    alignments: list[LocatorCandidate] = []
    early = 0
    # 三つ組の列挙は候補数の 3 乗。compact と micro は同じ引数で呼ぶので
    # 1 回だけ計算して使い回す (実測で 1 フレーム 31.6 回 -> 半減)。
    plain_triples: list | None = None
    for profile in profiles:
        if profile == "robust":
            # 2 重リングは一発識別できるので、候補を絞ったうえで
            # 三角形の条件を大きく緩められる (= 射影歪みに追従できる)。
            # 強い射影歪みで外側リングを取りこぼすことがあるので、
            # 絞った集合で組が作れなければ全候補にも当たる。
            ring = [c for c in cands if c.is_double_ring]
            triples = _equilateral_triples(ring, tol=0.9, pitch_tol=2.0, limit=8)
            if len(ring) < 3:
                continue  # 2 重リングが 3 つ揃わなければ robust ではない
            extra = _equilateral_triples(cands, tol=0.7, pitch_tol=1.5, limit=6)
            seen = {tuple(sorted((p.cx, p.cy) for p in t)) for t, _ in triples}
            triples += [(t, s) for t, s in extra
                        if tuple(sorted((p.cx, p.cy) for p in t)) not in seen]
            if triples:
                alignments = find_alignment_candidates(bz.dark)
                rep.alignments = len(alignments)
        else:
            if plain_triples is None:
                plain_triples = _equilateral_triples(cands)
            triples = plain_triples
        if not triples:
            continue
        rep.triples = max(rep.triples, len(triples))
        max_triples = budget[0] if budget else MAX_TRIPLES
        for pose in _pose_candidates(bz, triples, radius_hint, profile, alignments,
                                     max_triples):
            poses.append(pose)
            # 完全一致した姿勢はほぼ確実に正解。残りの半径 x 対応を評価する前に
            # 復号まで試してしまう。外れたら生成を続けるので堅牢性は落ちない。
            if pose.perfect and early < MAX_EARLY_ATTEMPTS:
                early += 1
                res = _try_pose(bz, pose, rep, scale)
                if res is not None:
                    return res

    if not poses:
        raise MutsumeError(
            f"姿勢候補がありません (ロケータ候補 {rep.candidates} 個, "
            f"三つ組 {rep.triples} 個)")

    poses.sort(key=lambda p: -p.score)
    # 試行回数はプロファイルごとに持つ。micro はシグネチャもフォーマットもなく
    # 姿勢候補に順位が付かない (すべて同点) ため多めに要るが、その枠が
    # compact / robust の探索を圧迫しないようにする。
    spent: dict[str, int] = {}
    for pose in poses:
        # 明らかに合っていない姿勢に復号までさせない。姿勢が外れていれば
        # シグネチャが崩れてスコアが大きく落ちるので、そこで切れる。
        # (見つからないフレームでは復号試行が 1 フレーム 50 回に達していた)
        if pose.score < POSE_TRY_MIN:
            continue
        cap = MAX_MICRO_ATTEMPTS if pose.profile == "micro" else MAX_FULL_ATTEMPTS
        if budget:
            cap = min(cap, budget[1])
        if spent.get(pose.profile, 0) >= cap:
            continue
        n = spent.get(pose.profile, 0) + 1
        spent[pose.profile] = n
        # 消失訂正の再試行は上位候補だけ。下位は姿勢自体が外れていることが多く、
        # どの消失集合でも通らないのに RS 復号を 3 倍払うことになる。
        res = _try_pose(bz, pose, rep, scale, erasures=n <= MAX_ERASURE_POSES)
        if res is not None:
            return res

    raise MutsumeError(
        f"{rep.attempts} 回試行しましたが復号できませんでした "
        f"(ロケータ候補 {rep.candidates} 個, 三つ組 {rep.triples} 個, "
        f"姿勢候補 {len(poses)} 個)"
    )


def _try_pose(bz: Binarized, pose: _Pose, rep: DetectionReport,
              scale: float, erasures: bool = True) -> DecodeResult | None:
    """1 つの姿勢でサンプリング -> 復号まで試す。駄目なら None。

    erasures=False なら消失訂正の再試行をしない。姿勢が怪しい候補では
    どの消失集合でも通らず、RS 復号を 3 倍払うだけになるため。
    """
    radius, profile = pose.radius, pose.profile
    for H, perspective in _pose_variants(bz, pose):
        sampled = sample_grid(bz, H, radius, profile)
        if sampled is None:
            continue
        rep.attempts += 1
        rep.perspective = perspective
        # フォーマット情報 (黒白のみ) からパレットを読み、データセルを
        # そのパレットで判定し直す。micro はフォーマット領域を持たず
        # 常に mono なのでそのまま。
        layout = get_layout(radius, profile)
        if profile != "micro":
            try:
                _, _, palette = read_format(layout, sampled.bits)
            except (KeyError, IndexError):
                continue
            sampled = sampled.with_palette(layout, palette)
            rep.palette = palette
        else:
            rep.palette = "mono"

        # まず素直に復号し、失敗したら確信度の低いセルを消失として再試行する
        tried: set[tuple[Axial, ...]] = set()
        for limit in (ERASURE_LIMITS if erasures else ERASURE_LIMITS[:1]):
            weak = tuple(sampled.weak_cells(limit)) if limit > 0 else ()
            if weak in tried:
                continue  # 同じ消失集合を二度試さない
            tried.add(weak)
            try:
                res = decode_grid(sampled.bits, radius, erase_cells=weak,
                                  profile=profile)
                res.perspective = perspective
                res.geometry = build_geometry(H, radius, profile, scale,
                                              perspective)
                rep.erasures = len(weak)
                rep.profile = profile
                rep.perspective = perspective
                return res
            except MutsumeError as e:
                rep.errors.append(f"{profile} r={radius} weak={len(weak)}: {e}")
    return None


def _pose_variants(bz: Binarized, pose: _Pose):
    """試す姿勢を順に返す。

    6 点ホモグラフィが既にある姿勢はそのまま使う。アフィン姿勢しかない場合
    (アライメントが blob 検出できなかったとき) は、予測 -> 局所探索の反復で
    ホモグラフィを起こす方法にフォールバックする。
    """
    yield pose.H, pose.perspective
    if pose.perspective or pose.profile == "compact":
        return
    refined = refine_homography(bz, pose.H, pose.radius, pose.profile)
    if refined is not None:
        yield refined, True


@lru_cache(maxsize=None)
def _triple_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C(n,3) の添字の組。候補数は高々 80 なのでキャッシュが効く。"""
    idx = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
    return idx[:, 0], idx[:, 1], idx[:, 2]


def _equilateral_triples(cands: list[LocatorCandidate], tol: float = TRIPLE_TOL,
                         pitch_tol: float = 0.5, limit: int = 16
                         ) -> list[tuple[tuple[LocatorCandidate, ...], float]]:
    """3 点が (ほぼ) 正三角形をなす組を、もっともらしい順に返す。

    tol / pitch_tol を緩めるとせん断や射影歪みにも追従する。ロケータが 2 重リングで
    一発識別できる robust プロファイルでは誤検出が少ないので大きく緩められる。

    さらに **辺長とセルピッチの整合** で足切りする。ロケータ 3 点の相互距離は
    必ず `sqrt(3) * ring * pitch` (ring = R-1 か R-2) なので、
    `辺長 / (sqrt(3) * pitch)` がありうるリング数の範囲に入らない組は本物になり得ない。
    画面や書類を撮ると UI の小さな白領域が大量に候補になり、それらが作る
    「大きいがピッチと辺長がまるで釣り合わない三角形」が上位を占めてしまうため。

    順位は 2 つの基準の上位を **混ぜて** 返す。

    1. リング数が整数に近い順 (整合性)。ピッチ推定の系統誤差は数 % なので、
       リング数が小さいうち (小さいシンボル) は整数判定がよく効く。
    2. 三角形が大きい順。リング数が大きいと誤差が 0.5 を超えて整数判定が
       無意味になるため、こちらが頼りになる。

    どちらか一方に寄せると、実写 (小さいシンボル + 画面の雑多な白領域) か
    レンダリング画像 (大きいシンボル) のどちらかで真の組を取りこぼす。
    """
    n = len(cands)
    if n < 3:
        return []
    # 組の列挙は n^3 / 6 (n=40 で約 1 万)。Python の三重比較ループだと
    # 失敗フレームで 1 フレーム 2 万組近く回るので、numpy で一括評価する。
    ia, ib, ic = _triple_indices(n)
    pts = np.array([(c.cx, c.cy) for c in cands], dtype=np.float64)
    pit = np.array([c.pitch for c in cands], dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))

    d3 = np.stack([dist[ia, ib], dist[ib, ic], dist[ic, ia]])  # (3, m)
    lo, hi = d3.min(axis=0), d3.max(axis=0)
    p3 = np.stack([pit[ia], pit[ib], pit[ic]])
    plo = p3.min(axis=0)
    side = d3.mean(axis=0)
    pitch = np.median(p3, axis=0)
    rings = np.divide(side, CELL_PITCH * pitch,
                      out=np.zeros_like(side), where=pitch > 0)
    keep = ((lo > 0) & ((hi - lo) <= tol * lo)
            & ((p3.max(axis=0) - plo) <= pitch_tol * plo)
            & (rings >= RING_RANGE[0]) & (rings <= RING_RANGE[1]))
    if not keep.any():
        return []

    # そのリング数で許される誤差で正規化した「整数からのずれ」
    r_k, s_k = rings[keep], side[keep]
    slack = np.maximum(0.2, r_k * PITCH_REL_TOL)
    plaus = np.minimum(1.0, np.abs(r_k - np.rint(r_k)) / slack)
    out = [(float(p), float(s), (cands[a], cands[b], cands[c]))
           for p, s, a, b, c in zip(plaus, s_k, ia[keep], ib[keep], ic[keep])]

    by_fit = sorted(out, key=lambda t: (t[0], -t[1]))
    by_size = sorted(out, key=lambda t: -t[1])
    merged: list[tuple[tuple[LocatorCandidate, ...], float]] = []
    seen: set[int] = set()
    for i in range(len(out)):
        for src in (by_fit, by_size):
            if i < len(src) and id(src[i][2]) not in seen:
                seen.add(id(src[i][2]))
                merged.append((src[i][2], src[i][1]))
        if len(merged) >= limit:
            break
    return merged[:limit]

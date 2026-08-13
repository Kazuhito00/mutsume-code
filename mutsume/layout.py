"""六角格子のジオメトリとシンボル配置 (機能セル / データ順序 / マスク)。

座標系: axial 座標 (q, r)。cube 座標は (x, y, z) = (q, -q-r, r)。
セル形状: pointy-top (尖り上) の六角形。したがってシンボル全体の外形は flat-top の
六角形になり、左右に頂点、上下に水平な辺が来る (デザイン画と同じ)。

シンボル構造 (radius = R):
  - セル総数 3R^2 + 3R + 1
  - ロケータ: リング R-1 のコーナー 0, 2, 4 を中心とする 7 セルの花形
      中心 = 白(0), 周囲 6 セル = 黒(1)
  - シグネチャ: リング R-1 のコーナー 1, 3, 5 から時計回りにずらした 6 セル (compact)
      -> 回転 60 度単位・鏡映のあいまいさを解消するための非対称マーカー
  - フォーマット: リング R の各ロケータ隣接位置に 6 ビット x 3 コピー = 18 セル (compact)
      内容 = ECC レベル 2 ビット + マスク ID 2 ビット + パレット 2 ビット (多数決で復元)
  - 残りすべてがデータセル (中心から外向きの螺旋順)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

Axial = tuple[int, int]

# axial の 6 方向。DIRS[i] * k がリング k のコーナー i になる。
DIRS: tuple[Axial, ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

MIN_RADIUS = 7  # R=6 はフォーマット領域が入らず、容量も 0
# RS はブロック分割するので 255 バイトの制限は外れている。R=40 で 4921 セル
# (610 バイト / 3 ブロック)。これ以上は描画・サンプリングのコストが実用外。
MAX_RADIUS = 40

# compact: コーナー 1/3/5 から 1,2 歩の 6 セル。コーナーからずらすことで
# パターンが鏡像に対して非対称になり、鏡映も判別できる。
SIGNATURE_OFFSETS = (1, 2)
SIGNATURE_VALUES = (1, 1,  # コーナー 1
                    1, 0,  # コーナー 3
                    0, 0)  # コーナー 5

# robust: コーナー 1/3/5 から 1,2,3 歩の 9 セル。3 セルだと偶然一致が 1/8 あり、
# 半径 x 対応付けを何百通りも試すと誤った姿勢が通ってしまう。9 セルで 1/512。
SIGNATURE_OFFSETS_ROBUST = (1, 2, 3)
SIGNATURE_VALUES_ROBUST = (1, 1, 0,  # コーナー 1
                           1, 0, 0,  # コーナー 3
                           0, 1, 0)  # コーナー 5
# フォーマット情報 6 ビット (ECC 2 + マスク 2 + パレット 2) を、
# リング R 上でロケータのコーナーを挟んだ両側に 3 セットずつ置く。
FORMAT_OFFSETS = (3, 4, 5, -3, -4, -5)

# マーカープロファイル
#   compact: ロケータ 7 セル x 3 + シグネチャ 6 + フォーマット 18 = 45 セル。
#            姿勢はアフィン変換まで (回転/拡縮/鏡映/せん断)。密度優先。
#   robust : ロケータ 19 セル x 3 (2 重リング) + アライメント 7 セル x 3
#            + シグネチャ 9 + フォーマット 6 = 93 セル。
#            ロケータを一発識別でき、6 点からホモグラフィを解いて射影歪みに対応。
#   micro  : ロケータ 7 セル x 3 = 21 セルのみ。シグネチャもフォーマットも持たず、
#            向き・半径・マスクは復号側の総当たり + CRC で決める。Micro QR 相当。
PROFILES = ("compact", "robust", "micro", "nano")
DEFAULT_PROFILE = "compact"
MIN_RADIUS_ROBUST = 9
MIN_RADIUS_MICRO = 5
MAX_RADIUS_MICRO = 8  # これ以上は総当たりが割に合わず compact の方が入る
# nano: 中央ブルズアイ 1 個 (白中心 + 黒リング = 7 セル) だけの最小規格。
# ロケータが 1 個なので姿勢は相似変換のみ (正対前提)。向きは回転 x 鏡映を総当たり。
MIN_RADIUS_NANO = 4
MAX_RADIUS_NANO = 6
NANO_BULLSEYE = (0, 1)


def hex_distance(a: Axial, b: Axial = (0, 0)) -> int:
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def hex_ring(k: int) -> list[Axial]:
    """リング k のセルをコーナー 0 から時計回りに列挙する。index i*k がコーナー i。"""
    if k == 0:
        return [(0, 0)]
    cells: list[Axial] = []
    q, r = DIRS[0][0] * k, DIRS[0][1] * k
    for i in range(6):
        d = DIRS[(i + 2) % 6]
        for _ in range(k):
            cells.append((q, r))
            q += d[0]
            r += d[1]
    return cells


def hex_spiral(radius: int) -> list[Axial]:
    """中心から外向きの螺旋順で全セルを列挙する。"""
    cells: list[Axial] = []
    for k in range(radius + 1):
        cells.extend(hex_ring(k))
    return cells


def rot60(c: Axial) -> Axial:
    """コーナー i -> コーナー i+1 となる 60 度回転。"""
    q, r = c
    return (q + r, -q)


def rot60_n(c: Axial, n: int) -> Axial:
    for _ in range(n % 6):
        c = rot60(c)
    return c


def mirror(c: Axial) -> Axial:
    """鏡映 (q, r) -> (r, q)。"""
    return (c[1], c[0])


def cell_count(radius: int) -> int:
    return 3 * radius * radius + 3 * radius + 1


def to_cartesian(c: Axial, size: float = 1.0) -> tuple[float, float]:
    """pointy-top セルの中心座標。size は外接円半径。"""
    q, r = c
    x = size * (3.0 ** 0.5) * (q + r / 2.0)
    y = size * 1.5 * r
    return x, y


CELL_PITCH = 3.0 ** 0.5  # size=1 のときの隣接セル中心間距離 (= セル幅)


# --- 機能セル -----------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    radius: int
    cells: tuple[Axial, ...]  # 螺旋順の全セル
    locator_centers: tuple[Axial, ...]  # コーナー 0, 2, 4 の順
    locator_cells: tuple[Axial, ...]  # 花形 7 セル x 3
    signature_cells: tuple[Axial, ...]  # 向き判定用の 3 セル
    function_values: dict[Axial, int]  # ロケータ + シグネチャの固定値
    format_positions: tuple[tuple[Axial, ...], ...]  # コピー x 4 ビット
    data_cells: tuple[Axial, ...]  # データ配置順
    profile: str = "compact"
    alignment_centers: tuple[Axial, ...] = ()  # robust のみ
    alignment_cells: tuple[Axial, ...] = ()

    @property
    def data_bit_capacity(self) -> int:
        return len(self.data_cells)

    @property
    def data_byte_capacity(self) -> int:
        return len(self.data_cells) // 8

    @property
    def marker_centers(self) -> tuple[Axial, ...]:
        """ロケータ 3 点 + アライメント 3 点。姿勢推定に使う対応点。"""
        return self.locator_centers + self.alignment_centers


def _add_disc(fixed: dict[Axial, int], center: Axial, values: tuple[int, ...],
              cell_set: set[Axial]) -> list[Axial]:
    """center を中心にリング 0..len(values)-1 を values[k] で塗る。"""
    used = []
    for k, v in enumerate(values):
        for c in hex_ring(k):
            cell = (center[0] + c[0], center[1] + c[1])
            assert cell in cell_set, f"marker cell {cell} outside symbol"
            assert cell not in fixed, f"marker cell {cell} overlaps another pattern"
            fixed[cell] = v
            used.append(cell)
    return used


@lru_cache(maxsize=None)
def get_layout(radius: int, profile: str = "compact") -> Layout:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    lo, hi = min_radius(profile), max_radius(profile)
    if not (lo <= radius <= hi):
        raise ValueError(f"radius must be in [{lo}, {hi}]: {radius}")

    cells = hex_spiral(radius)
    cell_set = set(cells)
    fixed: dict[Axial, int] = {}

    if profile == "nano":
        # 中央ブルズアイ 1 個だけ。シグネチャ・フォーマットは持たず総当たりで決める。
        bullseye = _add_disc(fixed, (0, 0), NANO_BULLSEYE, cell_set)
        data_cells = tuple(c for c in cells if c not in fixed)
        return Layout(
            radius=radius, cells=tuple(cells),
            locator_centers=((0, 0),), locator_cells=tuple(bullseye),
            signature_cells=(), function_values=fixed,
            format_positions=(), data_cells=data_cells, profile="nano",
        )

    if profile in ("compact", "micro"):
        # ロケータ = 花形 7 セル (白中心 + 黒リング)。リング R-1 のコーナー 0/2/4。
        ring_of_centers = radius - 1
        locator_values = (0, 1)
        alignment_values: tuple[int, ...] = ()
        # micro はシグネチャもフォーマットも持たない (総当たりで決める)
        format_corners = () if profile == "micro" else (0, 2, 4)
    else:
        # ロケータ = 2 重リング 19 セル (白中心 + 黒リング + 白リング)。
        # 外側の白リングまで含めて外縁に接するよう、中心はリング R-2 に置く。
        ring_of_centers = radius - 2
        locator_values = (0, 1, 0)
        alignment_values = (1, 0)  # 反転パターン: 黒中心 + 白リング
        format_corners = (0,)

    locator_centers: list[Axial] = []
    locator_cells: list[Axial] = []
    for corner in (0, 2, 4):
        center = (DIRS[corner][0] * ring_of_centers, DIRS[corner][1] * ring_of_centers)
        locator_centers.append(center)
        locator_cells.extend(_add_disc(fixed, center, locator_values, cell_set))

    alignment_centers: list[Axial] = []
    alignment_cells: list[Axial] = []
    if alignment_values:
        for corner in (1, 3, 5):
            center = (DIRS[corner][0] * ring_of_centers,
                      DIRS[corner][1] * ring_of_centers)
            alignment_centers.append(center)
            alignment_cells.extend(_add_disc(fixed, center, alignment_values, cell_set))

    ring = hex_ring(radius)
    n_ring = len(ring)

    # シグネチャ: コーナー 1/3/5 から時計回りに 1 歩ずらした位置。ずらすことで
    # パターンが鏡像に対して非対称になり、鏡映も判別できる。
    signature_cells: list[Axial] = []
    if profile == "micro":
        sig_positions: list[Axial] = []
        sig_values: tuple[int, ...] = ()
    elif profile == "compact":
        inner = hex_ring(radius - 1)
        sig_positions = [inner[(c * (radius - 1) + off) % len(inner)]
                         for c in (1, 3, 5) for off in SIGNATURE_OFFSETS]
        sig_values = SIGNATURE_VALUES
    else:
        sig_positions = [ring[(c * radius + off) % n_ring]
                         for c in (1, 3, 5) for off in SIGNATURE_OFFSETS_ROBUST]
        sig_values = SIGNATURE_VALUES_ROBUST
    for idx, cell in enumerate(sig_positions):
        assert cell not in fixed, "signature cell overlaps a marker"
        fixed[cell] = sig_values[idx]
        signature_cells.append(cell)

    # フォーマット領域: リング R 上、ロケータのコーナーから既定歩数の位置
    format_positions = []
    for corner in format_corners:
        base = corner * radius
        group = tuple(ring[(base + off) % n_ring] for off in FORMAT_OFFSETS)
        for cell in group:
            assert cell not in fixed, "format cell overlaps another pattern"
        format_positions.append(group)

    reserved = set(fixed) | {c for g in format_positions for c in g}
    data_cells = tuple(c for c in cells if c not in reserved)

    return Layout(
        radius=radius,
        cells=tuple(cells),
        locator_centers=tuple(locator_centers),
        locator_cells=tuple(locator_cells),
        signature_cells=tuple(signature_cells),
        function_values=fixed,
        format_positions=tuple(format_positions),
        data_cells=data_cells,
        profile=profile,
        alignment_centers=tuple(alignment_centers),
        alignment_cells=tuple(alignment_cells),
    )


def min_radius(profile: str) -> int:
    if profile == "robust":
        return MIN_RADIUS_ROBUST
    if profile == "micro":
        return MIN_RADIUS_MICRO
    if profile == "nano":
        return MIN_RADIUS_NANO
    return MIN_RADIUS


def max_radius(profile: str) -> int:
    if profile == "micro":
        return MAX_RADIUS_MICRO
    if profile == "nano":
        return MAX_RADIUS_NANO
    return MAX_RADIUS


# --- マスク -------------------------------------------------------------------

MASK_COUNT = 4


def mask_bit(mask_id: int, c: Axial) -> int:
    """データセルに XOR するマスク値。"""
    q, r = c
    s = -q - r
    if mask_id == 0:
        return (q + r) % 2
    if mask_id == 1:
        return 1 if (q - r) % 3 == 0 else 0
    if mask_id == 2:
        return ((q * r) % 2 + (q + s) % 3) % 2
    if mask_id == 3:
        return ((q + 2 * r) % 3 + (s % 2)) % 2
    raise ValueError(f"unknown mask id: {mask_id}")


def mask_penalty(grid: dict[Axial, int], layout: Layout) -> int:
    """マスク選択用のペナルティ。黒白の偏り + 同色の直線連続を罰する。"""
    values = [grid[c] for c in layout.cells]
    dark = sum(values)
    total = len(values)
    penalty = abs(dark * 2 - total) * 2  # 偏り

    # 3 方向の直線に沿った同色ランを評価
    for d in (DIRS[0], DIRS[1], DIRS[2]):
        seen: set[Axial] = set()
        for start in layout.cells:
            prev = (start[0] - d[0], start[1] - d[1])
            if prev in grid or start in seen:
                continue
            run_val = -1
            run_len = 0
            cur = start
            while cur in grid:
                seen.add(cur)
                v = grid[cur]
                if v == run_val:
                    run_len += 1
                else:
                    if run_len >= 4:
                        penalty += 3 + (run_len - 4) * 2
                    run_val = v
                    run_len = 1
                cur = (cur[0] + d[0], cur[1] + d[1])
            if run_len >= 4:
                penalty += 3 + (run_len - 4) * 2
    return penalty

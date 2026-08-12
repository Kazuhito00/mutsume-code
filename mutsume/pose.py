"""シンボルの姿勢（画像座標との対応）。

デコード時に求めたホモグラフィをそのまま持ち回り、
「ファインダはどこか」「シンボルの外形はどこか」「任意のセルはどの画素か」を
元画像の座標系で答えられるようにする。

座標はすべて **入力画像そのままのピクセル座標**。デコーダは長辺 1400px に
縮小して処理するが、その縮小率は打ち消してから返す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .layout import DIRS, Axial, get_layout, to_cartesian


def project(H: np.ndarray, cell: Axial) -> tuple[float, float]:
    """セル (axial) の中心を画像座標へ射影する。"""
    return project_point(H, to_cartesian(cell, 1.0))


def project_point(H: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    d = H[2, 0] * xy[0] + H[2, 1] * xy[1] + H[2, 2]
    if abs(d) < 1e-12:
        d = 1e-12
    # numpy スカラーではなく素の float で返す (API から numpy 型を漏らさない)
    return (float((H[0, 0] * xy[0] + H[0, 1] * xy[1] + H[0, 2]) / d),
            float((H[1, 0] * xy[0] + H[1, 1] * xy[1] + H[1, 2]) / d))


def project_xy(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """モデル座標 (N,2) を画像座標 (N,2) へ射影する。"""
    ones = np.ones((len(xy), 1))
    p = np.hstack([xy, ones]) @ H.T
    w = p[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return p[:, :2] / w


Point = tuple[float, float]


@dataclass
class Geometry:
    """復号したシンボルの画像上の位置。座標は入力画像のピクセル単位。"""

    radius: int
    profile: str
    homography: np.ndarray  # 3x3: モデル座標 -> 入力画像座標
    finders: tuple[Point, ...]  # ファインダ (ロケータ) 中心 x3
    alignments: tuple[Point, ...]  # アライメント中心 x3 (robust のみ)
    outline: tuple[Point, ...]  # シンボル外形の 6 頂点
    center: Point
    cell_size: float  # 隣接セル中心間の距離 (px)
    rotation_deg: float  # コーナー 0 方向の角度 (画像 x 軸から時計回り)
    mirrored: bool  # 鏡映 (裏返し・鏡越し) に見えているか
    perspective: bool  # ホモグラフィ (射影) で求めたか

    # --- 便利メソッド ---------------------------------------------------

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """外形の外接矩形 (x0, y0, x1, y1)。"""
        xs = [p[0] for p in self.outline]
        ys = [p[1] for p in self.outline]
        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    @property
    def corner0(self) -> Point:
        """コーナー 0 (シンボルの「正面」方向) の外形頂点。

        シグネチャ (micro は CRC) が 6 回転 x 鏡映の曖昧さを解いているので、
        復号できた時点で向きは一意に決まっている。outline[0] がその頂点。
        """
        return self.outline[0]

    def scaled(self, k: float) -> "Geometry":
        """座標を k 倍したコピーを返す。

        デコードは縮小した作業画像で行うことが多いので、別解像度のフレームに
        重ね描きするときに使う。回転・鏡映はスケール不変なのでそのまま。
        """
        if k == 1.0:
            return self

        def s(pts):
            return tuple((x * k, y * k) for x, y in pts)

        return Geometry(
            radius=self.radius, profile=self.profile,
            homography=np.diag([k, k, 1.0]) @ self.homography,
            finders=s(self.finders), alignments=s(self.alignments),
            outline=s(self.outline),
            center=(self.center[0] * k, self.center[1] * k),
            cell_size=self.cell_size * k, rotation_deg=self.rotation_deg,
            mirrored=self.mirrored, perspective=self.perspective)

    def cell_center(self, cell: Axial) -> Point:
        """任意のセルの中心座標。"""
        return project(self.homography, cell)

    def cell_polygon(self, cell: Axial) -> tuple[Point, ...]:
        """任意のセルの六角形頂点 (6 点)。"""
        cx, cy = to_cartesian(cell, 1.0)
        pts = []
        for i in range(6):
            a = math.radians(60 * i - 90)
            pts.append(project_point(self.homography,
                                     (cx + math.cos(a), cy + math.sin(a))))
        return tuple(pts)

    def cell_centers(self) -> dict[Axial, Point]:
        """全セルの中心座標。"""
        cells = get_layout(self.radius, self.profile).cells
        model = np.array([to_cartesian(c, 1.0) for c in cells], dtype=np.float64)
        pts = project_xy(self.homography, model)
        return {c: (float(p[0]), float(p[1])) for c, p in zip(cells, pts)}

    def to_dict(self, ndigits: int = 2) -> dict:
        """JSON にしやすい素の dict。座標は ndigits 桁に丸める。"""
        def r(v) -> float:
            return round(float(v), ndigits)

        def rp(pts) -> list[list[float]]:
            return [[r(x), r(y)] for x, y in pts]

        return {
            "radius": self.radius,
            "profile": self.profile,
            "finders": rp(self.finders),
            "alignments": rp(self.alignments),
            "outline": rp(self.outline),
            "center": [r(self.center[0]), r(self.center[1])],
            "bbox": [r(v) for v in self.bbox],
            "cell_size": r(self.cell_size),
            "rotation_deg": r(self.rotation_deg),
            "mirrored": self.mirrored,
            "perspective": self.perspective,
            "homography": [[round(float(v), 6) for v in row]
                           for row in self.homography],
        }


def build_geometry(H: np.ndarray, radius: int, profile: str, scale: float,
                   perspective: bool) -> Geometry:
    """作業画像で求めたホモグラフィから、入力画像座標の Geometry を作る。

    scale は「入力画像 -> 作業画像」の縮小率。座標を戻すため打ち消す。
    """
    if scale <= 0:
        scale = 1.0
    H_full = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ H

    layout = get_layout(radius, profile)
    finders = tuple(project(H_full, c) for c in layout.locator_centers)
    alignments = tuple(project(H_full, c) for c in layout.alignment_centers)

    # 外形: リング R のコーナーセルの、さらに外側の頂点。
    # モデル座標では中心から (R + 0.5) / R 倍の位置になる。
    k = (radius + 0.5) / radius
    outline = tuple(
        project_point(H_full, tuple(v * k for v in to_cartesian(
            (DIRS[i][0] * radius, DIRS[i][1] * radius), 1.0)))
        for i in range(6))

    center = project(H_full, (0, 0))
    nbr = project(H_full, (1, 0))
    cell_size = float(math.dist(center, nbr))
    corner0 = project(H_full, (DIRS[0][0] * radius, DIRS[0][1] * radius))
    rotation = float(math.degrees(math.atan2(corner0[1] - center[1],
                                             corner0[0] - center[0])))
    # 鏡映: 中心でのヤコビアンの向き (基底ベクトルの外積の符号) で判定する
    ey = project_point(H_full, (0.0, 1.0))
    mirrored = ((nbr[0] - center[0]) * (ey[1] - center[1])
                - (nbr[1] - center[1]) * (ey[0] - center[0])) < 0

    return Geometry(radius=radius, profile=profile, homography=H_full,
                    finders=finders, alignments=alignments, outline=outline,
                    center=center, cell_size=cell_size,
                    rotation_deg=rotation, mirrored=mirrored,
                    perspective=perspective)

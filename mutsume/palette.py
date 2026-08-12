"""白黒パレット (mono, 1 bit/セル)。

palette 機構自体は残してあり (フォーマット情報にパレット 2 ビットを持つ)、
将来カラーを足せるようにしてある。マーカー (ロケータ / アライメント /
シグネチャ / フォーマット) もデータセルもすべて黒白で描く。
"""

from __future__ import annotations

PALETTES: dict[str, tuple[tuple[float, float, float], ...]] = {
    # 1 bit: 0 = 白, 1 = 黒 (機能セルの値と同じ規約: 1 が暗い)
    "mono": ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
}

PALETTE_NAMES = tuple(PALETTES)  # フォーマット情報に入れる 2 ビットの並び順
DEFAULT_PALETTE = "mono"

BITS_PER_CELL = {"mono": 1}


def bits_per_cell(palette: str) -> int:
    try:
        return BITS_PER_CELL[palette]
    except KeyError:
        raise ValueError(f"unknown palette: {palette}") from None


def palette_index(palette: str) -> int:
    return PALETTE_NAMES.index(palette)


def palette_by_index(idx: int) -> str:
    return PALETTE_NAMES[idx % len(PALETTE_NAMES)]


def rgb255(palette: str, value: int) -> tuple[int, int, int]:
    c = PALETTES[palette][value]
    return (round(c[0] * 255), round(c[1] * 255), round(c[2] * 255))


def classify_batch(palette: str, rgb: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """(N,3) の正規化 RGB を最も近いパレット色に一括分類し、(値, 確信度) を返す。

    確信度 = (第 2 候補との距離 - 最良距離) / 色間の最小距離。
    1.0 に近いほど自信があり、0 に近いほど紛らわしい。
    セルごとに判定すると復号試行 1 回あたり数百回の Python 呼び出しになるため、
    まとめて距離行列で解く。
    """
    import numpy as np

    entries = np.asarray(PALETTES[palette], dtype=np.float64)  # (k, 3)
    d = np.linalg.norm(rgb[:, None, :] - entries[None, :, :], axis=2)  # (N, k)
    idx = np.argmin(d, axis=1)
    if entries.shape[0] < 2:
        return idx, np.ones(len(rgb))
    part = np.partition(d, 1, axis=1)
    conf = np.clip((part[:, 1] - part[:, 0]) / _min_separation(palette), 0.0, 1.0)
    return idx, conf


_SEPARATION_CACHE: dict[str, float] = {}


def _min_separation(palette: str) -> float:
    if palette not in _SEPARATION_CACHE:
        entries = PALETTES[palette]
        best = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                d = sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5
                if best is None or d < best:
                    best = d
        _SEPARATION_CACHE[palette] = best or 1.0
    return _SEPARATION_CACHE[palette]

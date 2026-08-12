"""シンボルの描画 (PNG / SVG / コンソール)。"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .codec import Symbol
from .layout import DIRS, Axial, to_cartesian

DARK = (17, 17, 17)
LIGHT = (255, 255, 255)
BACKGROUND = (247, 245, 240)
OUTLINE = (17, 17, 17)
SEPARATOR = (255, 255, 255)  # 黒セル同士の境界に入れる白抜き線

# hex_vertices の辺 k (頂点 k -> k+1) に接する隣接セルの方向。
EDGE_DIRS: tuple[Axial, ...] = (DIRS[1], DIRS[0], DIRS[5], DIRS[4], DIRS[3], DIRS[2])


def hex_vertices(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    """pointy-top 六角形の頂点 (上が尖る)。"""
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 90)
        pts.append((cx + size * math.cos(ang), cy + size * math.sin(ang)))
    return pts


def cell_fill(symbol: Symbol, cell: Axial,
              dark: tuple[int, int, int] = DARK,
              light: tuple[int, int, int] = LIGHT) -> tuple[int, int, int]:
    """セルの塗り色。値が 1 (暗) なら dark、0 (明) なら light。"""
    return dark if symbol.grid[cell] else light


def _is_format_cell(symbol: Symbol, cell: Axial) -> bool:
    return any(cell in group for group in symbol.layout.format_positions)


def _is_dark(rgb: tuple[int, int, int]) -> bool:
    """境界線の描き分け用の明暗判定 (知覚輝度)。"""
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 128


def _edge_color(here: bool, there: bool | None, dark_separator: bool,
                grid_lines: bool):
    """セル間 (there=None なら外周) の境界線の色。描かないなら None。

    here / there は「暗いセルか」。
    """
    if there is None:
        # 外周: 明るいセルは輪郭を描く。暗いセルは背景との明度差で足りる。
        return "out" if (grid_lines and not here) else None
    if here and there:
        return "sep" if dark_separator else None
    return "out" if grid_lines else None


def _collect_edges(symbol: Symbol, verts: dict[Axial, list[tuple[float, float]]],
                   dark_separator: bool, grid_lines: bool,
                   dark=DARK, light=LIGHT):
    """描画すべき境界線を (白セパレータ, 黒輪郭) に分けて返す。

    白セパレータを先に、黒輪郭を後から重ねて描く。セル頂点を最終的に黒が占め、
    白セルの輪郭リングが途切れないようにするため。

    セパレータはロケータ花形の内部にも引く (意匠を統一する)。このため花形の
    中心セルは白セパレータ経由で外側の白と繋がるが、デコーダ側は白マスクを
    収縮させてから連結成分を取るので細線は切断され、検出には影響しない。
    """
    dark_of = {c: _is_dark(cell_fill(symbol, c, dark, light))
               for c in symbol.layout.cells}
    seps: list[tuple[tuple[float, float], tuple[float, float]]] = []
    outs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for cell in symbol.layout.cells:
        pts = verts[cell]
        here = dark_of[cell]
        for k, d in enumerate(EDGE_DIRS):
            nb = (cell[0] + d[0], cell[1] + d[1])
            there = dark_of.get(nb)
            if there is not None and nb < cell:
                continue  # 共有辺は片方からだけ描く
            kind = _edge_color(here, there, dark_separator, grid_lines)
            if kind == "sep":
                seps.append((pts[k], pts[(k + 1) % 6]))
            elif kind == "out":
                outs.append((pts[k], pts[(k + 1) % 6]))
    return seps, outs


def _geometry(symbol: Symbol, cell_size: float, quiet_cells: float):
    xs, ys = [], []
    for cell in symbol.layout.cells:
        x, y = to_cartesian(cell, cell_size)
        xs.append(x)
        ys.append(y)
    margin = cell_size * (3.0 ** 0.5) * quiet_cells + cell_size
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    width = int(math.ceil(max_x - min_x))
    height = int(math.ceil(max_y - min_y))
    return -min_x, -min_y, width, height


def render_png(
    symbol: Symbol,
    path: str | None = None,
    cell_size: float = 18.0,
    quiet_cells: float = 1.5,
    dark_separator: bool = True,
    grid_lines: bool = True,
    line_ratio: float = 0.06,
    supersample: int = 3,
    dark_rgb: tuple[int, int, int] = DARK,
    light_rgb: tuple[int, int, int] = LIGHT,
) -> Image.Image:
    """シンボルを PNG 画像として描画する。

    dark_separator=True のとき、隣り合う黒セルの間に白抜きの境界線を入れる
    (黒がベタ塗りに融合せず、1 セルずつ視認できるようになる)。
    """
    ox, oy, width, height = _geometry(symbol, cell_size, quiet_cells)
    ss = max(1, supersample)
    img = Image.new("RGB", (width * ss, height * ss), light_rgb)
    draw = ImageDraw.Draw(img)
    stroke = max(1, int(round(cell_size * line_ratio * ss)))

    # 1) 塗り
    verts: dict[Axial, list[tuple[float, float]]] = {}
    for cell in symbol.layout.cells:
        x, y = to_cartesian(cell, cell_size)
        pts = [((px + ox) * ss, (py + oy) * ss) for px, py in
               hex_vertices(x, y, cell_size)]
        verts[cell] = pts
        draw.polygon(pts, fill=cell_fill(symbol, cell, dark_rgb, light_rgb))

    # 2) 境界線: 白セパレータ -> 黒輪郭 の順に描く
    seps, outs = _collect_edges(symbol, verts, dark_separator, grid_lines,
                                dark_rgb, light_rgb)
    cap = stroke / 2.0
    for color, segs in ((light_rgb, seps), (dark_rgb, outs)):
        for p1, p2 in segs:
            draw.line([p1, p2], fill=color, width=stroke)
            if stroke > 2:  # 角に隙間ができないよう丸キャップを足す
                for x, y in (p1, p2):
                    draw.ellipse([x - cap, y - cap, x + cap, y + cap], fill=color)

    if ss > 1:
        img = img.resize((width, height), Image.LANCZOS)
    if path:
        img.save(path)
    return img


def render_svg(symbol: Symbol, path: str | None = None, cell_size: float = 18.0,
               quiet_cells: float = 1.5, dark_separator: bool = True,
               grid_lines: bool = True, line_ratio: float = 0.06,
               dark_rgb: tuple[int, int, int] = DARK,
               light_rgb: tuple[int, int, int] = LIGHT) -> str:
    ox, oy, width, height = _geometry(symbol, cell_size, quiet_cells)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="rgb{light_rgb}"/>',
    ]
    sw = cell_size * line_ratio
    verts: dict[Axial, list[tuple[float, float]]] = {}
    for cell in symbol.layout.cells:
        x, y = to_cartesian(cell, cell_size)
        pts = [(px + ox, py + oy) for px, py in hex_vertices(x, y, cell_size)]
        verts[cell] = pts
        poly = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        parts.append(f'<polygon points="{poly}" fill="rgb{cell_fill(symbol, cell, dark_rgb, light_rgb)}"/>')

    seps, outs = _collect_edges(symbol, verts, dark_separator, grid_lines,
                                dark_rgb, light_rgb)
    for color, segs in ((light_rgb, seps), (dark_rgb, outs)):
        for (x1, y1), (x2, y2) in segs:
            parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
                         f'y2="{y2:.2f}" stroke="rgb{color}" '
                         f'stroke-width="{sw:.2f}" stroke-linecap="round"/>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
    return svg


# 向き矢印の長さ (中心 -> コーナー 0 の何割まで伸ばすか)。webcamera_demo と共有
ORIENTATION_ARROW_SHRINK = 0.8


def draw_geometry(image: Image.Image, geometry, path: str | None = None,
                  width: int | None = None) -> Image.Image:
    """検出した位置を元画像に重ねて描く (検証・デバッグ用)。

    外形を緑、ファインダを赤、アライメントを青、中心を十字で示す。
    """
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w = width or max(2, int(round(geometry.cell_size * 0.12)))
    r = max(3.0, geometry.cell_size * 0.42)

    d.polygon([tuple(p) for p in geometry.outline], outline=(0, 200, 0), width=w)
    for p in geometry.outline:  # 角の欠けを埋める
        d.ellipse([p[0] - w, p[1] - w, p[0] + w, p[1] + w], fill=(0, 200, 0))

    for cx, cy in geometry.finders:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(230, 30, 30), width=w)
    for cx, cy in geometry.alignments:
        d.ellipse([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7],
                  outline=(30, 80, 230), width=w)

    cx, cy = geometry.center
    d.line([cx - r, cy, cx + r, cy], fill=(230, 140, 0), width=w)
    d.line([cx, cy - r, cx, cy + r], fill=(230, 140, 0), width=w)

    # 向き: 中心からコーナー 0 へ矢印 (先端に丸)。シンボルの回転が一目で分かる
    tx, ty = geometry.corner0
    k = ORIENTATION_ARROW_SHRINK
    ex, ey = cx + (tx - cx) * k, cy + (ty - cy) * k
    d.line([cx, cy, ex, ey], fill=(230, 140, 0), width=w)
    tip = max(3.0, geometry.cell_size * 0.2)
    d.ellipse([ex - tip, ey - tip, ex + tip, ey + tip], fill=(230, 140, 0))

    if path:
        out.save(path)
    return out


_TEXT_GLYPHS = ".#*+oxOX"


def render_text(symbol: Symbol, dark: str = "#", light: str = ".") -> str:
    """デバッグ用のコンソール表示。列位置 = 2q + r で蜂の巣状に並べる。"""
    cells: dict[Axial, int] = symbol.grid
    cols = {2 * q + r for q, r in cells}
    min_col = min(cols)
    rows: dict[int, dict[int, int]] = {}
    for (q, r), v in cells.items():
        rows.setdefault(r, {})[2 * q + r - min_col] = v
    mono = symbol.palette == "mono"
    lines = []
    for r in sorted(rows):
        row = rows[r]
        buf = [" "] * (max(row) + 1)
        for col, v in row.items():
            buf[col] = (dark if v else light) if mono else _TEXT_GLYPHS[v & 7]
        lines.append("".join(buf))
    return "\n".join(lines)

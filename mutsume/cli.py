"""コマンドラインインタフェース。

    python -m mutsume encode "HELLO" -o out.png
    python -m mutsume decode out.png
    python -m mutsume info
"""

from __future__ import annotations

import argparse
import json
import sys

from .codec import (
    ECC_LEVELS,
    MutsumeError,
    block_plan,
    cell_capacity_bytes,
    encode,
    micro_parity,
    parity_count,
    payload_capacity,
)
from .palette import DEFAULT_PALETTE, PALETTE_NAMES
from .detect import DetectionReport, decode_image
from .layout import (
    DEFAULT_PROFILE,
    PROFILES,
    get_layout,
    max_radius,
    min_radius,
)
def _cmd_encode(args: argparse.Namespace) -> int:
    data = sys.stdin.buffer.read() if args.text == "-" else args.text
    sym = encode(data, ecc=args.ecc, radius=args.radius, mask=args.mask,
                 profile=args.profile, palette=args.palette)
    out = args.output
    sym.save(out, cell_size=args.cell_size, quiet_cells=args.quiet,
             dark_separator=not args.no_separator,
             grid_lines=not args.no_grid_lines)
    print(f"wrote {out}  profile={sym.profile} palette={sym.palette} "
          f"radius={sym.radius} ecc={sym.ecc} mask={sym.mask_id} "
          f"cells={len(sym.layout.cells)} segments={sym.segments}")
    if args.show:
        print(sym.to_text())
    return 0


def _cmd_decode(args: argparse.Namespace) -> int:
    rep = DetectionReport()
    try:
        res = decode_image(args.image, radius_hint=args.radius, report=rep,
                           profile=args.profile)
    except MutsumeError as e:
        print(f"decode failed: {e}", file=sys.stderr)
        if args.verbose:
            for line in rep.errors[:10]:
                print("  " + line, file=sys.stderr)
        return 1
    print(res.text if not args.raw else res.payload)

    if (args.geometry or args.overlay) and res.geometry is not None:
        g = res.geometry
        if args.geometry:
            print(json.dumps(g.to_dict(), ensure_ascii=False, indent=2))
        if args.overlay:
            from PIL import Image

            from .render import draw_geometry
            draw_geometry(Image.open(args.image), g, args.overlay)
            print(f"# overlay -> {args.overlay}", file=sys.stderr)

    if args.verbose:
        print(f"# profile={res.profile} palette={res.palette} "
              f"radius={res.radius} ecc={res.ecc} "
              f"mask={res.mask_id} corrected={res.errors_corrected} "
              f"erasures={res.erasures_used} perspective={res.perspective} "
              f"binarization={rep.binarization} segments={res.segments} "
              f"bytes={len(res.payload)}", file=sys.stderr)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    mode, profile, palette = args.mode, args.profile, args.palette
    lo, hi = min_radius(profile), max_radius(profile)

    if profile == "micro":
        print(f"モード={mode} プロファイル=micro の格納可能量 "
              f"(ECC 固定 / mono 固定)")
        print(f"{'R':>3} {'cells':>6} {'data':>5} {'nsym':>5} {'容量':>6}")
        for r in range(lo, hi + 1):
            lay = get_layout(r, profile)
            total = len(lay.data_cells) // 8
            try:
                ns = str(micro_parity(total))
                cap = str(payload_capacity(r, "M", mode, profile))
            except MutsumeError:
                ns, cap = "-", "-"
            print(f"{r:>3} {len(lay.cells):>6} {total:>5} {ns:>5} {cap:>6}")
        return 0

    print(f"モード={mode} プロファイル={profile} パレット={palette} "
          f"の格納可能量 (ECC レベル別)")
    print(f"{'R':>3} {'cells':>6} {'data':>5} {'blk':>3} " +
          " ".join(f"{lv:>6}" for lv in ECC_LEVELS))
    radii = [r for r in range(lo, hi + 1)
             if r <= 20 or r % 2 == 0 or r == hi]
    for r in radii:
        lay = get_layout(r, profile)
        total = cell_capacity_bytes(lay, palette)
        caps = []
        nblk = 1
        for lv in ECC_LEVELS:
            try:
                caps.append(str(payload_capacity(r, lv, mode, profile, palette)))
                nblk = len(block_plan(total, parity_count(total, lv)))
            except MutsumeError:
                caps.append("-")
        print(f"{r:>3} {len(lay.cells):>6} {total:>5} {nblk:>3} " +
              " ".join(f"{c:>6}" for c in caps))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mutsume", description="hexagonal 2D code (PoC)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("encode", help="テキストをシンボル画像にする")
    e.add_argument("text", help="埋め込む文字列 ('-' で標準入力)")
    e.add_argument("-o", "--output", default="mutsume.png", help="出力ファイル (.png/.svg)")
    e.add_argument("--ecc", default="M", choices=list(ECC_LEVELS))
    e.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(PROFILES),
                   help="compact=マーカー最小(アフィンまで) / robust=射影変換対応")
    e.add_argument("--palette", default=DEFAULT_PALETTE, choices=list(PALETTE_NAMES),
                   help="白黒 (mono)")
    e.add_argument("--radius", type=int, default=None, help="シンボル半径 (省略で自動)")
    e.add_argument("--mask", type=int, default=None, help="マスク ID (省略で自動選択)")
    e.add_argument("--cell-size", type=float, default=18.0)
    e.add_argument("--quiet", type=float, default=1.5, help="クワイエットゾーン (セル数)")
    e.add_argument("--no-separator", action="store_true",
                   help="隣接する黒セル間の白抜き境界線を入れない")
    e.add_argument("--no-grid-lines", action="store_true",
                   help="白セルの黒い輪郭線を描かない")
    e.add_argument("--show", action="store_true", help="ASCII でも表示する")
    e.set_defaults(func=_cmd_encode)

    d = sub.add_parser("decode", help="画像からシンボルを読み取る")
    d.add_argument("image")
    d.add_argument("--radius", type=int, default=None, help="半径が既知なら指定")
    d.add_argument("--profile", default=None, choices=list(PROFILES),
                   help="省略時は両方試す")
    d.add_argument("--raw", action="store_true", help="bytes のまま表示")
    d.add_argument("--geometry", action="store_true",
                   help="ファインダ位置・外形などを JSON で出力")
    d.add_argument("--overlay", metavar="FILE",
                   help="検出位置を元画像に重ねた確認用画像を書き出す")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=_cmd_decode)

    i = sub.add_parser("info", help="サイズ別の容量表を表示")
    i.add_argument("--mode", default="byte", choices=["byte", "alnum", "numeric"],
                   help="容量を表示する文字モード")
    i.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(PROFILES))
    i.add_argument("--palette", default=DEFAULT_PALETTE, choices=list(PALETTE_NAMES))
    i.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

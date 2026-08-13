"""論理層のエンコーダ / デコーダ。

RS 符号語の中身 (仕様バージョン 0x02):
    byte0        : 仕様バージョン
    byte1..2     : ビットストリーム長 (uint16 BE)
    byte3..      : ビットストリーム (bits.py のセグメント列)
    +2 バイト    : CRC-16/CCITT-FALSE (byte0 からビットストリーム末尾まで)
    残り         : パディング (0xEC, 0x11 の繰り返し)

上記全体を複数の Reed-Solomon ブロックに分割し、インターリーブして
データセルに並べる。1 ブロックは 255 バイト以下。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from .bits import (
    MODE_ALNUM,
    MODE_BITS,
    MODE_BYTE,
    MODE_NAMES,
    MODE_NUMERIC,
    SWITCH_BITS,
    BitReader,
    BitWriter,
    choose_segments,
    decode_segments,
    encode_segments,
    read_segment_data,
    segment_bits,
    widest_mode,
    write_segment_data,
)
from .layout import (
    Axial,
    DEFAULT_PROFILE,
    Layout,
    MASK_COUNT,
    MAX_RADIUS,
    PROFILES,
    get_layout,
    mask_bit,
    mask_penalty,
    max_radius,
    min_radius,
    mirror,
    rot60_n,
)
from .palette import (
    DEFAULT_PALETTE,
    PALETTE_NAMES,
    bits_per_cell,
    palette_by_index,
    palette_index,
)
from .rs import RSDecodeError, rs_decode, rs_encode

if TYPE_CHECKING:  # 循環 import を避ける (pose は layout にしか依存しない)
    from .pose import Geometry

SPEC_VERSION = 0x02
HEADER_LEN = 3
CRC_LEN = 2
OVERHEAD = HEADER_LEN + CRC_LEN
PAD_BYTES = (0xEC, 0x11)
MAX_BLOCK = 255  # RS 符号語 1 ブロックの上限

# ECC レベル -> 符号語に占めるパリティの割合
ECC_LEVELS = ("L", "M", "Q", "H")
ECC_RATIO = {"L": 0.15, "M": 0.25, "Q": 0.35, "H": 0.45}


class MutsumeError(Exception):
    pass


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# --- micro プロファイル -------------------------------------------------------
#
# 小さいシンボルでは固定オーバーヘッドが容量を食い尽くす。R=7 の compact は
# データ 15B のうち 機能セル 45 + RS 4B + ヘッダ/CRC 5B が乗り、数字 9 桁しか
# 入らない。micro はここを削る:
#
#   * 機能セルはロケータ 21 セルのみ (シグネチャ / フォーマット領域なし)
#   * ECC レベルは固定 (約 25%、下限 2 バイト) なので signalling 不要
#   * マスク ID と向きと半径は復号側で総当たりし、CRC-16 で正解を選ぶ
#   * ヘッダは モード 2 bit + 個数 6 bit の 1 バイトのみ (仕様版・長さ 2B を廃止)
#   * CRC-16 は符号語データ部の末尾 2 バイトに固定配置 (位置が既知なので長さ不要)
#
# 代償: ECC レベルを選べない、セグメント分割なし (1 モードのみ)、
#       復号コストが増える (半径 x 向き x マスク の総当たり)。
MICRO_MODE_BITS = 2
MICRO_COUNT_BITS = 6
MICRO_MAX_COUNT = (1 << MICRO_COUNT_BITS) - 1
MICRO_CRC_LEN = 2
MICRO_HEADER_LEN = 1
MICRO_OVERHEAD = MICRO_HEADER_LEN + MICRO_CRC_LEN
MICRO_MIN_PARITY = 2
MICRO_RATIO = 0.25
MICRO_MODES = (MODE_NUMERIC, MODE_ALNUM, MODE_BYTE)


def micro_parity(total_bytes: int) -> int:
    """micro の RS パリティ長。半径から一意に決まるので符号内に書かない。"""
    nsym = max(MICRO_MIN_PARITY, 2 * round(total_bytes * MICRO_RATIO / 2))
    if nsym + MICRO_OVERHEAD >= total_bytes:
        raise MutsumeError("micro フォーマットにはシンボルが小さすぎます")
    return nsym


def micro_capacity(radius: int, mode: str = "byte", profile: str = "micro") -> int:
    """micro / nano で格納できる文字数 / バイト数。"""
    if mode not in MODE_BY_NAME:
        raise MutsumeError(f"不明なモード: {mode}")
    total = len(get_layout(radius, profile).data_cells) // 8
    # ビットストリームに使えるのは (データ符号語 - CRC 2B)。
    # そこからヘッダ (モード 2bit + 個数 6bit) を引く。
    budget = (total - micro_parity(total) - MICRO_CRC_LEN) * 8 - \
        MICRO_MODE_BITS - MICRO_COUNT_BITS
    if budget <= 0:
        return 0
    m = MODE_BY_NAME[mode]
    n = budget // 8
    while n < MICRO_MAX_COUNT and segment_bits(m, n + 1) <= budget:
        n += 1
    return n


def _build_micro_body(payload: bytes, k_total: int) -> bytes:
    mode = widest_mode(payload)
    if len(payload) > MICRO_MAX_COUNT:
        raise MutsumeError(
            f"micro に入るのは最大 {MICRO_MAX_COUNT} 単位です ({len(payload)} 単位)")
    w = BitWriter()
    w.write(MICRO_MODES.index(mode), MICRO_MODE_BITS)
    w.write(len(payload), MICRO_COUNT_BITS)
    write_segment_data(w, mode, payload)
    stream = w.to_bytes()
    room = k_total - MICRO_CRC_LEN
    if len(stream) > room:
        raise MutsumeError(
            f"micro に収まりません ({len(stream)} バイト必要, 上限 {room} バイト)")
    out = bytearray(stream)
    i = 0
    while len(out) < room:
        out.append(PAD_BYTES[i % 2])
        i += 1
    out += crc16(bytes(out)).to_bytes(MICRO_CRC_LEN, "big")
    return bytes(out)


def _parse_micro_body(body: bytes) -> tuple[bytes, list[tuple[str, int]]]:
    if len(body) <= MICRO_CRC_LEN:
        raise MutsumeError("micro の符号語が短すぎます")
    head, want = body[:-MICRO_CRC_LEN], body[-MICRO_CRC_LEN:]
    if crc16(head) != int.from_bytes(want, "big"):
        raise MutsumeError("CRC が一致しません")
    r = BitReader(head)
    idx = r.read(MICRO_MODE_BITS)
    if idx >= len(MICRO_MODES):
        raise MutsumeError(f"不明な micro モード: {idx}")
    mode = MICRO_MODES[idx]
    count = r.read(MICRO_COUNT_BITS)
    try:
        payload = read_segment_data(r, mode, count)
    except ValueError as e:
        raise MutsumeError(f"micro セグメントの復号に失敗しました: {e}") from e
    return payload, [(MODE_NAMES[mode], count)]


def parity_count(total_bytes: int, ecc: str) -> int:
    nsym = int(round(total_bytes * ECC_RATIO[ecc]))
    nsym = max(4, nsym - nsym % 2)  # 偶数にして t = nsym/2 を明確にする
    if nsym >= total_bytes:
        raise MutsumeError("このシンボルサイズには ECC レベルが高すぎます")
    return nsym


# --- RS ブロック分割 ----------------------------------------------------------


def block_plan(total: int, nsym_total: int) -> list[tuple[int, int]]:
    """符号語全体を (データ長, パリティ長) のブロックに割る。

    ブロック数は 255 バイト制限から決まる最小値。各ブロックのパリティ比率が
    全体と揃うように、余りを先頭のブロックから 1 バイトずつ配る。
    """
    if nsym_total >= total:
        raise MutsumeError("パリティが大きすぎてデータ領域が残りません")
    nblocks = max(1, -(-total // MAX_BLOCK))
    sizes = [total // nblocks + (1 if i < total % nblocks else 0)
             for i in range(nblocks)]
    parities = [nsym_total // nblocks + (1 if i < nsym_total % nblocks else 0)
                for i in range(nblocks)]
    plan = []
    for size, par in zip(sizes, parities):
        if par >= size:
            raise MutsumeError("ブロックのパリティが大きすぎてデータ領域が残りません")
        plan.append((size - par, par))
    return plan


def interleave_order(plan: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """ストリーム位置 -> (ブロック番号, ブロック内の符号語位置) の対応。

    データ部を全ブロック横断で 1 バイトずつ、続いてパリティ部も同様に並べる。
    連続した汚れが 1 ブロックに集中せず、全ブロックへ均等に散る。
    """
    ks = [k for k, _ in plan]
    ss = [s for _, s in plan]
    order: list[tuple[int, int]] = []
    for i in range(max(ks)):
        for b, k in enumerate(ks):
            if i < k:
                order.append((b, i))
    for i in range(max(ss)):
        for b, s in enumerate(ss):
            if i < s:
                order.append((b, ks[b] + i))
    return order


def _encode_blocks(body: bytes, plan: list[tuple[int, int]]) -> bytes:
    blocks = []
    pos = 0
    for k, s in plan:
        blocks.append(rs_encode(body[pos:pos + k], s))
        pos += k
    order = interleave_order(plan)
    return bytes(blocks[b][i] for b, i in order)


def _decode_blocks(stream: bytes, plan: list[tuple[int, int]],
                   erase_pos: Iterable[int] = ()) -> tuple[bytes, int]:
    order = interleave_order(plan)
    if len(stream) != len(order):
        raise MutsumeError("符号語長がブロック構成と一致しません")
    blocks: list[list[int]] = [[0] * (k + s) for k, s in plan]
    erasures: list[list[int]] = [[] for _ in plan]
    for j, (b, i) in enumerate(order):
        blocks[b][i] = stream[j]
    for j in erase_pos:
        if 0 <= j < len(order):
            b, i = order[j]
            erasures[b].append(i)

    body = bytearray()
    corrected = 0
    for idx, (k, s) in enumerate(plan):
        try:
            fixed, n = rs_decode(blocks[idx], s, erase_pos=erasures[idx] or None)
        except RSDecodeError as e:
            raise MutsumeError(f"ブロック {idx}: {e}") from e
        body += fixed[:k]
        corrected += n
    return bytes(body), corrected


def data_capacity(radius: int, ecc: str, profile: str = DEFAULT_PROFILE,
                  palette: str = DEFAULT_PALETTE) -> int:
    """ヘッダ / CRC を除いた、ビットストリームに使えるバイト数。"""
    total = cell_capacity_bytes(get_layout(radius, profile), palette)
    nsym = parity_count(total, ecc)
    plan = block_plan(total, nsym)
    return sum(k for k, _ in plan) - OVERHEAD


MODE_BY_NAME = {"numeric": MODE_NUMERIC, "alnum": MODE_ALNUM, "byte": MODE_BYTE}


def payload_capacity(radius: int, ecc: str, mode: str = "byte",
                     profile: str = DEFAULT_PROFILE,
                     palette: str = DEFAULT_PALETTE) -> int:
    """指定モード 1 セグメントで格納できるペイロードの文字数 / バイト数。"""
    if mode not in MODE_BY_NAME:
        raise MutsumeError(f"不明なモード: {mode}")
    if profile in ("micro", "nano"):
        # micro / nano は ECC 固定・mono 固定
        return micro_capacity(radius, mode, profile)
    # ビットストリームに使えるビット数から、セグメントヘッダと終端を引く
    budget = data_capacity(radius, ecc, profile, palette) * 8 - SWITCH_BITS - MODE_BITS
    if budget <= 0:
        return 0
    m = MODE_BY_NAME[mode]
    n = budget // 8  # どのモードでも 8 bit/単位 以下なので下界になる
    while segment_bits(m, n + 1) <= budget:
        n += 1
    return n


@dataclass
class Symbol:
    radius: int
    ecc: str
    mask_id: int
    grid: dict[Axial, int]
    segments: list[tuple[str, int]] = field(default_factory=list)
    profile: str = DEFAULT_PROFILE
    palette: str = DEFAULT_PALETTE

    @property
    def layout(self) -> Layout:
        return get_layout(self.radius, self.profile)

    def bit(self, cell: Axial) -> int:
        return self.grid[cell]

    # 描画はメソッドとして提供する (公開 API を encode/decode に絞るため。
    # オプションは mutsume.render の各関数と同じ)。

    @staticmethod
    def _render():
        # 遅延 import: 循環 import (render -> codec) を避ける
        from . import render
        return render

    def to_image(self, **options):
        """PIL の Image として描画する。"""
        return self._render().render_png(self, None, **options)

    def to_svg(self, **options) -> str:
        """SVG 文字列として描画する。"""
        return self._render().render_svg(self, None, **options)

    def to_text(self, **options) -> str:
        """コンソール確認用の ASCII アート。"""
        return self._render().render_text(self, **options)

    def save(self, path: str, **options) -> None:
        """画像ファイルに書き出す。拡張子が .svg なら SVG、それ以外は PNG。"""
        r = self._render()
        write = (r.render_svg if str(path).lower().endswith(".svg")
                 else r.render_png)
        write(self, str(path), **options)


# --- 符号語の組み立て ---------------------------------------------------------


def _build_body(stream: bytes, k_total: int) -> bytes:
    body = bytes([SPEC_VERSION]) + len(stream).to_bytes(2, "big") + stream
    body += crc16(body).to_bytes(2, "big")
    if len(body) > k_total:
        raise MutsumeError(f"データが収まりません ({len(body)} バイト必要, 上限 {k_total} バイト)")
    i = 0
    out = bytearray(body)
    while len(out) < k_total:
        out.append(PAD_BYTES[i % 2])
        i += 1
    return bytes(out)


def _parse_body(body: bytes) -> bytes:
    if len(body) < OVERHEAD:
        raise MutsumeError("符号語が短すぎます")
    if body[0] != SPEC_VERSION:
        raise MutsumeError(f"不明な仕様バージョン: 0x{body[0]:02x}")
    length = int.from_bytes(body[1:3], "big")
    end = HEADER_LEN + length
    if end + CRC_LEN > len(body):
        raise MutsumeError("長さフィールドが符号語を超えています")
    head = body[:end]
    want = int.from_bytes(body[end:end + CRC_LEN], "big")
    if crc16(head) != want:
        raise MutsumeError("CRC が一致しません")
    return body[HEADER_LEN:end]


# --- ビット配置 ---------------------------------------------------------------


def cell_capacity_bytes(layout: Layout, palette: str) -> int:
    """データセルに載る総バイト数 (パレットのビット数ぶん増える)。"""
    return len(layout.data_cells) * bits_per_cell(palette) // 8


def _place_values(layout: Layout, codeword: bytes, palette: str) -> dict[Axial, int]:
    k = bits_per_cell(palette)
    bits = []
    for byte in codeword:
        bits.extend((byte >> s) & 1 for s in range(7, -1, -1))
    bits.extend([0] * (len(layout.data_cells) * k - len(bits)))  # 端数
    out = {}
    for i, cell in enumerate(layout.data_cells):
        v = 0
        for b in bits[i * k:(i + 1) * k]:
            v = (v << 1) | b
        out[cell] = v
    return out


def _cell_values(layout: Layout, grid: dict[Axial, int]) -> "np.ndarray":
    """データセルの値を螺旋順の配列で取り出す。"""
    return np.fromiter((grid[c] for c in layout.data_cells), dtype=np.uint8,
                       count=len(layout.data_cells))


def _pack_values(values: "np.ndarray", nbytes: int, palette: str) -> bytes:
    """セル値の配列 (データセル順) を MSB first で nbytes に詰める。

    復号の試行 1 回ごとに呼ばれる (失敗フレームで 50 回) ので、
    セルごとの Python ループではなく packbits で一括にする。
    """
    k = bits_per_cell(palette)
    if k == 1:
        bits = values & 1
    else:
        shifts = np.arange(k - 1, -1, -1, dtype=np.uint8)
        bits = ((values[:, None] >> shifts[None, :]) & 1).reshape(-1)
    return np.packbits(bits[:nbytes * 8]).tobytes()


FORMAT_BITS = 6


def _format_bits(ecc: str, mask_id: int, palette: str) -> list[int]:
    v = ((ECC_LEVELS.index(ecc) << 4) | ((mask_id & 0b11) << 2)
         | (palette_index(palette) & 0b11))
    return [(v >> s) & 1 for s in range(FORMAT_BITS - 1, -1, -1)]


def _apply_function_cells(layout: Layout, grid: dict[Axial, int], ecc: str,
                          mask_id: int, palette: str) -> None:
    grid.update(layout.function_values)
    fbits = _format_bits(ecc, mask_id, palette)
    for group in layout.format_positions:
        for cell, bit in zip(group, fbits):
            grid[cell] = bit


def read_format(layout: Layout, grid: dict[Axial, int]) -> tuple[str, int, str]:
    """複製されたフォーマット情報をビットごとの多数決で読む。"""
    groups = layout.format_positions
    v = 0
    for i in range(FORMAT_BITS):
        votes = sum(1 for group in groups if grid[group[i]] & 1)
        v = (v << 1) | (1 if votes * 2 >= len(groups) else 0)
    return (ECC_LEVELS[(v >> 4) & 0b11], (v >> 2) & 0b11,
            palette_by_index(v & 0b11))


def check_signature(layout: Layout, grid: dict[Axial, int]) -> bool:
    """ロケータ / シグネチャの固定セルが期待値と一致するか (向き判定の高速フィルタ)。"""
    return all(grid.get(cell) == val for cell, val in layout.function_values.items())


# --- 公開 API -----------------------------------------------------------------


def encode(
    payload: bytes | str,
    ecc: str = "M",
    radius: int | None = None,
    mask: int | None = None,
    profile: str = DEFAULT_PROFILE,
    palette: str = DEFAULT_PALETTE,
) -> Symbol:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if ecc not in ECC_RATIO:
        raise MutsumeError(f"不明な ECC レベル: {ecc}")
    if profile not in PROFILES:
        raise MutsumeError(f"不明なプロファイル: {profile}")
    if palette not in PALETTE_NAMES:
        raise MutsumeError(f"不明なパレット: {palette}")

    if profile in ("micro", "nano"):
        return _encode_micro(payload, radius, mask, profile)

    segments = choose_segments(payload)
    stream = encode_segments(payload, segments)
    need = len(stream) + OVERHEAD

    if radius is None:
        radius = _smallest_radius(need, ecc, profile, palette)
    layout = get_layout(radius, profile)
    total = cell_capacity_bytes(layout, palette)
    nsym = parity_count(total, ecc)
    plan = block_plan(total, nsym)
    k_total = sum(k for k, _ in plan)
    codeword = _encode_blocks(_build_body(stream, k_total), plan)

    base = _place_values(layout, codeword, palette)
    candidates = range(MASK_COUNT) if mask is None else [mask]
    best = None
    for m in candidates:
        grid = {c: v ^ mask_value(m, c, palette) for c, v in base.items()}
        _apply_function_cells(layout, grid, ecc, m, palette)
        score = mask_penalty(grid, layout)
        if best is None or score < best[0]:
            best = (score, m, grid)
    assert best is not None
    _, mask_id, grid = best

    return Symbol(radius=radius, ecc=ecc, mask_id=mask_id, grid=grid,
                  segments=_segment_info(payload, segments), profile=profile,
                  palette=palette)


def _encode_micro(payload: bytes, radius: int | None, mask: int | None,
                  profile: str = "micro") -> Symbol:
    """micro / nano の符号化。ECC 固定・単一セグメント・1 RS ブロック。"""
    mode_name = MODE_NAMES[widest_mode(payload)]
    if radius is None:
        for r in range(min_radius(profile), max_radius(profile) + 1):
            try:
                if micro_capacity(r, mode_name, profile) >= len(payload):
                    radius = r
                    break
            except MutsumeError:
                continue
        if radius is None:
            raise MutsumeError(
                f"{profile} には大きすぎます ({len(payload)} 単位, 最大 "
                f"{micro_capacity(max_radius(profile), mode_name, profile)} 単位)。"
                "profile='compact' を使ってください")

    layout = get_layout(radius, profile)
    total = len(layout.data_cells) // 8
    nsym = micro_parity(total)
    codeword = rs_encode(_build_micro_body(payload, total - nsym), nsym)

    base = _place_values(layout, codeword, DEFAULT_PALETTE)
    best = None
    for m in (range(MASK_COUNT) if mask is None else [mask]):
        grid = {c: v ^ mask_bit(m, c) for c, v in base.items()}
        grid.update(layout.function_values)
        score = mask_penalty(grid, layout)
        if best is None or score < best[0]:
            best = (score, m, grid)
    assert best is not None
    _, mask_id, grid = best
    return Symbol(radius=radius, ecc="-", mask_id=mask_id, grid=grid,
                  segments=[(mode_name, len(payload))], profile=profile,
                  palette=DEFAULT_PALETTE)


def _decode_micro(grid: dict[Axial, int], radius: int,
                  profile: str = "micro") -> DecodeResult:
    """マスク ID を総当たりして復号する (micro / nano はフォーマット領域を持たない)。"""
    layout = get_layout(radius, profile)
    total = len(layout.data_cells) // 8
    nsym = micro_parity(total)
    errors = []
    values = _cell_values(layout, grid)
    for mask_id in range(MASK_COUNT):
        tbl = mask_table(radius, profile, DEFAULT_PALETTE, mask_id)
        raw = _pack_values(values ^ tbl, total, DEFAULT_PALETTE)
        try:
            body, nerr = rs_decode(raw, nsym)
        except RSDecodeError as e:
            errors.append(f"mask={mask_id}: {e}")
            continue
        try:
            payload, info = _parse_micro_body(body[:total - nsym])
        except MutsumeError as e:
            errors.append(f"mask={mask_id}: {e}")
            continue
        return DecodeResult(payload=payload, radius=radius, ecc="-",
                            mask_id=mask_id, errors_corrected=nerr,
                            segments=info, profile=profile)
    raise MutsumeError(f"{profile} の復号に失敗しました: " + "; ".join(errors[:2]))


@lru_cache(maxsize=None)
def mask_table(radius: int, profile: str, palette: str, mask_id: int) -> "np.ndarray":
    """データセル並び順のマスク値。姿勢候補ごとに毎回計算すると効かないので覚える。"""
    cells = get_layout(radius, profile).data_cells
    return np.fromiter((mask_value(mask_id, c, palette) for c in cells),
                       dtype=np.uint8, count=len(cells))


def mask_value(mask_id: int, cell: Axial, palette: str) -> int:
    """セル値に XOR するマスク。パレットのビット数だけ独立な面を重ねる。"""
    v = 0
    for i in range(bits_per_cell(palette)):
        v = (v << 1) | mask_bit((mask_id + i) % MASK_COUNT, cell)
    return v


def _segment_info(payload: bytes, segments) -> list[tuple[str, int]]:
    names = {MODE_NUMERIC: "numeric", MODE_ALNUM: "alnum", MODE_BYTE: "byte"}
    return [(names[m], e - s) for m, s, e in segments]


def _smallest_radius(need_bytes: int, ecc: str, profile: str, palette: str) -> int:
    for r in range(min_radius(profile), MAX_RADIUS + 1):
        try:
            if data_capacity(r, ecc, profile, palette) + OVERHEAD >= need_bytes:
                return r
        except MutsumeError:
            continue
    raise MutsumeError(
        f"データが大きすぎます ({need_bytes} バイト必要, 最大 "
        f"{data_capacity(MAX_RADIUS, ecc, profile, palette) + OVERHEAD} バイト)")


@dataclass
class DecodeResult:
    payload: bytes
    radius: int
    ecc: str
    mask_id: int
    errors_corrected: int
    erasures_used: int = 0
    segments: list[tuple[str, int]] = field(default_factory=list)
    profile: str = DEFAULT_PROFILE
    palette: str = DEFAULT_PALETTE
    orientation: int = 0  # 0..5 = 60 度回転数
    mirrored: bool = False
    perspective: bool = False  # ホモグラフィで姿勢を求めたか
    # 画像から読んだ場合のみ設定される。ファインダ位置・外形・ホモグラフィなど
    geometry: "Geometry | None" = None

    @property
    def text(self) -> str:
        return self.payload.decode("utf-8", errors="replace")


def erasure_bytes(layout: Layout, cells: Iterable[Axial],
                  palette: str = DEFAULT_PALETTE) -> list[int]:
    """壊れているセルの集合を、RS 符号語のバイト位置に変換する。

    データセルは螺旋順に k ビットずつ詰まっているので、1 セルは最大 2 バイトに
    またがる。またがる場合は両方を消失にする。
    """
    k = bits_per_cell(palette)
    index = {c: i for i, c in enumerate(layout.data_cells)}
    total = cell_capacity_bytes(layout, palette)
    out = set()
    for c in cells:
        i = index.get(c)
        if i is None:
            continue
        for b in range(i * k // 8, (i * k + k - 1) // 8 + 1):
            if b < total:  # 末尾の端数は符号語外
                out.add(b)
    return sorted(out)


def decode_grid(grid: dict[Axial, int], radius: int,
                erase_cells: Iterable[Axial] | None = None,
                profile: str = DEFAULT_PROFILE) -> DecodeResult:
    """向きが確定済みの格子から復号する。

    erase_cells に「読み取りが怪しいセル」を渡すと、そのセルを含むバイトを
    RS の消失位置として扱う。誤り e + 消失 f は 2e + f <= nsym まで訂正できるため、
    位置が分かっている汚れに対しては訂正能力が実質 2 倍になる。
    """
    layout = get_layout(radius, profile)
    missing = [c for c in layout.cells if c not in grid]
    if missing:
        raise MutsumeError(f"グリッドに {len(missing)} 個のセルが欠けています")

    if profile in ("micro", "nano"):
        return _decode_micro(grid, radius, profile)

    ecc, mask_id, palette = read_format(layout, grid)
    total = cell_capacity_bytes(layout, palette)
    nsym = parity_count(total, ecc)
    plan = block_plan(total, nsym)

    tbl = mask_table(radius, profile, palette, mask_id)
    raw = _pack_values(_cell_values(layout, grid) ^ tbl, total, palette)
    erase_pos = erasure_bytes(layout, erase_cells, palette) if erase_cells else []
    if len(erase_pos) > nsym:
        raise MutsumeError(f"消失が多すぎます ({len(erase_pos)} > {nsym})")

    body, nerr = _decode_blocks(raw, plan, erase_pos)
    stream = _parse_body(body)
    try:
        payload, info = decode_segments(stream)
    except ValueError as e:
        raise MutsumeError(f"セグメントの復号に失敗しました: {e}") from e
    return DecodeResult(payload=payload, radius=radius, ecc=ecc, mask_id=mask_id,
                        errors_corrected=nerr, erasures_used=len(erase_pos),
                        segments=info, profile=profile, palette=palette)


def orient_grid(grid: dict[Axial, int], rot: int, mirrored: bool) -> dict[Axial, int]:
    """観測格子 obs から、回転 rot / 鏡映を打ち消した格子を作る。"""
    out = {}
    for cell in grid:
        src = rot60_n(cell, rot)
        if mirrored:
            src = mirror(src)
        if src in grid:
            out[cell] = grid[src]
    return out


def decode_grid_any_orientation(grid: dict[Axial, int], radius: int,
                                profile: str = DEFAULT_PROFILE) -> DecodeResult:
    """6 回転 x 鏡映を総当たりし、シグネチャ + CRC が通る向きを採用する。"""
    layout = get_layout(radius, profile)
    errors = []
    ordered = []
    for mir in (False, True):
        for rot in range(6):
            cand = orient_grid(grid, rot, mir)
            if len(cand) != len(layout.cells):
                continue
            score = 0 if check_signature(layout, cand) else 1
            ordered.append((score, rot, mir, cand))
    ordered.sort(key=lambda t: t[0])
    for score, rot, mir, cand in ordered:
        try:
            res = decode_grid(cand, radius, profile=profile)
        except MutsumeError as e:
            errors.append(str(e))
            continue
        res.orientation = rot
        res.mirrored = mir
        return res
    raise MutsumeError("どの向きでも正しく復号できませんでした: " + "; ".join(errors[:3]))


def decode(symbol: Symbol) -> DecodeResult:
    return decode_grid(symbol.grid, symbol.radius, profile=symbol.profile)

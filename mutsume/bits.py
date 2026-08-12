"""ビット列の読み書きと、文字モード別のセグメント符号化。

モード (3 ビット) + 文字数 (12 ビット) + データ、を並べたビットストリームを作る。

  END     (0b000) 終端
  NUMERIC (0b001) 数字のみ。3 桁 -> 10 ビット (端数 2 桁 -> 7、1 桁 -> 4)
  ALNUM   (0b010) 45 文字集合。2 文字 -> 11 ビット (端数 1 文字 -> 6)
  BYTE    (0b011) 生バイト。1 バイト -> 8 ビット

数字は 3.33 ビット/文字、英数は 5.5 ビット/文字となり、バイトモード (8) に比べて
実効容量が 2.4 倍 / 1.45 倍になる。
"""

from __future__ import annotations

MODE_BITS = 3
COUNT_BITS = 12
MAX_SEGMENT_UNITS = (1 << COUNT_BITS) - 1

MODE_END = 0b000
MODE_NUMERIC = 0b001
MODE_ALNUM = 0b010
MODE_BYTE = 0b011

MODE_NAMES = {MODE_NUMERIC: "numeric", MODE_ALNUM: "alnum", MODE_BYTE: "byte"}

ALNUM_CHARS = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
ALNUM_INDEX = {c: i for i, c in enumerate(ALNUM_CHARS)}
DIGITS = set(b"0123456789")

# セグメント切り替えの固定コスト (ビット)
SWITCH_BITS = MODE_BITS + COUNT_BITS


class BitWriter:
    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, value: int, n: int) -> None:
        if value >> n:
            raise ValueError(f"value {value} does not fit in {n} bits")
        self._bits.extend((value >> s) & 1 for s in range(n - 1, -1, -1))

    def __len__(self) -> int:
        return len(self._bits)

    def to_bytes(self) -> bytes:
        bits = self._bits
        pad = (-len(bits)) % 8
        out = bytearray()
        acc = 0
        for i, b in enumerate(bits + [0] * pad):
            acc = (acc << 1) | b
            if i % 8 == 7:
                out.append(acc)
                acc = 0
        return bytes(out)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def remaining(self) -> int:
        return len(self._data) * 8 - self._pos

    def read(self, n: int) -> int:
        if n > self.remaining:
            raise ValueError("bit stream exhausted")
        v = 0
        for _ in range(n):
            byte = self._data[self._pos >> 3]
            v = (v << 1) | ((byte >> (7 - (self._pos & 7))) & 1)
            self._pos += 1
        return v


# --- モード判定 ---------------------------------------------------------------


def _classes(data: bytes) -> list[int]:
    """各バイトが収まる最も狭いモードを返す。"""
    out = []
    for b in data:
        if b in DIGITS:
            out.append(MODE_NUMERIC)
        elif b in ALNUM_INDEX:
            out.append(MODE_ALNUM)
        else:
            out.append(MODE_BYTE)
    return out


def _fits(mode: int, cls: int) -> bool:
    """cls のバイトが mode で符号化できるか (NUMERIC ⊂ ALNUM ⊂ BYTE)。"""
    if mode == MODE_BYTE:
        return True
    if mode == MODE_ALNUM:
        return cls in (MODE_NUMERIC, MODE_ALNUM)
    return cls == MODE_NUMERIC


def segment_bits(mode: int, n: int) -> int:
    """mode で n 単位を符号化したときのデータ部ビット数。"""
    if mode == MODE_NUMERIC:
        return 10 * (n // 3) + (0, 4, 7)[n % 3]
    if mode == MODE_ALNUM:
        return 11 * (n // 2) + 6 * (n % 2)
    return 8 * n


def _runs(classes: list[int]) -> list[tuple[int, int, int]]:
    """(開始, 終了, クラス) の連続領域に分解する。"""
    runs = []
    i = 0
    while i < len(classes):
        j = i
        while j < len(classes) and classes[j] == classes[i]:
            j += 1
        runs.append((i, j, classes[i]))
        i = j
    return runs


def choose_segments(data: bytes, max_runs: int = 96) -> list[tuple[int, int, int]]:
    """(mode, start, end) のセグメント列を選ぶ。総ビット数最小になるよう DP で探索。

    連続領域 (run) の境界でのみ切るので探索は O(runs^2)。run が多すぎる場合は
    切り替えコストの方が大きいのでバイトモード 1 本にまとめる。
    """
    if not data:
        return []
    classes = _classes(data)
    runs = _runs(classes)
    if len(runs) > max_runs:
        return _split_long(MODE_BYTE, 0, len(data))

    n = len(runs)
    INF = float("inf")
    best: list[float] = [INF] * (n + 1)
    choice: list[tuple[int, int] | None] = [None] * (n + 1)
    best[n] = 0.0
    for a in range(n - 1, -1, -1):
        for mode in (MODE_NUMERIC, MODE_ALNUM, MODE_BYTE):
            for b in range(a + 1, n + 1):
                if not _fits(mode, runs[b - 1][2]):
                    break  # 以降の run も収まらない
                start, end = runs[a][0], runs[b - 1][1]
                cost = SWITCH_BITS + segment_bits(mode, end - start) + best[b]
                if cost < best[a]:
                    best[a] = cost
                    choice[a] = (mode, b)

    segments: list[tuple[int, int, int]] = []
    a = 0
    while a < n:
        assert choice[a] is not None
        mode, b = choice[a]
        segments.append((mode, runs[a][0], runs[b - 1][1]))
        a = b

    # 4095 単位を超えるセグメントは分割する
    out: list[tuple[int, int, int]] = []
    for mode, s, e in segments:
        out.extend(_split_long(mode, s, e))
    return out


def _split_long(mode: int, start: int, end: int) -> list[tuple[int, int, int]]:
    out = []
    while end - start > MAX_SEGMENT_UNITS:
        cut = start + MAX_SEGMENT_UNITS
        if mode == MODE_NUMERIC:
            cut -= (cut - start) % 3  # 端数が出ないよう 3 の倍数で切る
        elif mode == MODE_ALNUM:
            cut -= (cut - start) % 2
        out.append((mode, start, cut))
        start = cut
    out.append((mode, start, end))
    return out


# --- 符号化 / 復号 ------------------------------------------------------------


def write_segment_data(w: BitWriter, mode: int, chunk: bytes) -> None:
    """モード別のデータ本体を書く (モード / 個数のヘッダは呼び出し側)。"""
    if mode == MODE_NUMERIC:
        for i in range(0, len(chunk) - 2, 3):
            w.write(int(chunk[i:i + 3]), 10)
        rest = len(chunk) % 3
        if rest:
            w.write(int(chunk[-rest:]), (0, 4, 7)[rest])
    elif mode == MODE_ALNUM:
        for i in range(0, len(chunk) - 1, 2):
            w.write(ALNUM_INDEX[chunk[i]] * 45 + ALNUM_INDEX[chunk[i + 1]], 11)
        if len(chunk) % 2:
            w.write(ALNUM_INDEX[chunk[-1]], 6)
    else:
        for b in chunk:
            w.write(b, 8)


def read_segment_data(r: BitReader, mode: int, count: int) -> bytes:
    out = bytearray()
    if mode == MODE_NUMERIC:
        for _ in range(count // 3):
            out += f"{r.read(10):03d}".encode()
        rest = count % 3
        if rest:
            out += f"{r.read((0, 4, 7)[rest]):0{rest}d}".encode()
    elif mode == MODE_ALNUM:
        for _ in range(count // 2):
            v = r.read(11)
            out.append(ALNUM_CHARS[v // 45])
            out.append(ALNUM_CHARS[v % 45])
        if count % 2:
            out.append(ALNUM_CHARS[r.read(6)])
    else:
        for _ in range(count):
            out.append(r.read(8))
    return bytes(out)


def widest_mode(data: bytes) -> int:
    """データ全体を 1 セグメントで表せる最も狭いモード。"""
    cls = _classes(data)
    if not cls:
        return MODE_NUMERIC
    if all(c == MODE_NUMERIC for c in cls):
        return MODE_NUMERIC
    if all(c in (MODE_NUMERIC, MODE_ALNUM) for c in cls):
        return MODE_ALNUM
    return MODE_BYTE


def encode_segments(data: bytes, segments: list[tuple[int, int, int]]) -> bytes:
    w = BitWriter()
    for mode, start, end in segments:
        chunk = data[start:end]
        w.write(mode, MODE_BITS)
        w.write(len(chunk), COUNT_BITS)
        write_segment_data(w, mode, chunk)
    w.write(MODE_END, MODE_BITS)
    return w.to_bytes()


def decode_segments(stream: bytes) -> tuple[bytes, list[tuple[str, int]]]:
    """ビットストリームを復号し、(データ, [(モード名, 単位数)]) を返す。"""
    r = BitReader(stream)
    out = bytearray()
    info: list[tuple[str, int]] = []
    while r.remaining >= MODE_BITS:
        mode = r.read(MODE_BITS)
        if mode == MODE_END:
            break
        if mode not in MODE_NAMES:
            raise ValueError(f"unknown segment mode: {mode}")
        count = r.read(COUNT_BITS)
        info.append((MODE_NAMES[mode], count))
        out += read_segment_data(r, mode, count)
    return bytes(out), info


def encoded_size(data: bytes) -> int:
    """データをビットストリームにしたときのバイト数。"""
    return len(encode_segments(data, choose_segments(data)))

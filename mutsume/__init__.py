"""mutsume-code: 六角格子ベースの 2 次元コード。

    import mutsume

    sym = mutsume.encode("HELLO")                # -> Symbol
    sym.save("hello.png")

    res = mutsume.decode("hello.png")            # -> DecodeResult
    print(res.text, res.geometry.center)

    for res in mutsume.decode_all("photo.jpg", max_symbols=4):
        print(res.text, res.geometry.bbox)

    mutsume.capacity(radius=10)                  # 何文字入るかの見積もり

公開 API は上の 4 関数と、その返り値の型 (Symbol / DecodeResult / Geometry)、
例外 MutsumeError のみ。描画オプションの詳細・検出の内部・容量表などは
mutsume.render / mutsume.detect / mutsume.codec / mutsume.layout から使える。
"""

from .codec import DecodeResult, MutsumeError, Symbol, encode, payload_capacity
from .codec import decode as _decode_symbol
from .detect import decode_image as _decode_image
from .detect import decode_image_all as _decode_image_all
from .layout import DEFAULT_PROFILE as _DEFAULT_PROFILE
from .palette import DEFAULT_PALETTE as _DEFAULT_PALETTE
from .pose import Geometry

__all__ = [
    "DecodeResult",
    "Geometry",
    "MutsumeError",
    "Symbol",
    "capacity",
    "decode",
    "decode_all",
    "encode",
    "__version__",
]

__version__ = "0.2.0"


def decode(source, **options) -> DecodeResult:
    """コードを 1 つ読み取る。読めなければ MutsumeError。

    source には画像 (ファイルパス / PIL.Image) か、encode() が返した Symbol を
    渡せる。画像のときのオプションは mutsume.detect.decode_image と同じ
    (radius_hint / profile / report / exclude)。
    """
    if isinstance(source, Symbol):
        return _decode_symbol(source)
    return _decode_image(source, **options)


def decode_all(source, **options) -> list[DecodeResult]:
    """画像に写っているコードをすべて読み取る (0 個なら空リスト)。

    動画では hints に前フレームの `DecodeResult.geometry` を渡すと
    トラッキングになる。オプション (max_symbols=8 / hints / hints_only など) は
    mutsume.detect.decode_image_all と同じで、既定値もそちらに従う。
    """
    return _decode_image_all(source, **options)


def capacity(radius: int, ecc: str = "M", mode: str = "byte",
             profile: str = _DEFAULT_PROFILE,
             palette: str = _DEFAULT_PALETTE) -> int:
    """指定サイズに入るペイロードの文字数 / バイト数を返す。

    mode は "numeric" (数字) / "alnum" (英数 45 文字) / "byte"。
    """
    return payload_capacity(radius, ecc, mode, profile, palette)

"""いろいろなパターンで encode -> 画像保存 -> decode を行い、往復を検証するテスト。

生成した PNG / SVG は tests/generated/ に残すので、目視でも確認できる。

    venv\\Scripts\\python.exe -m unittest tests.test_roundtrip_images -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mutsume import MutsumeError, decode, decode_all, encode  # noqa: E402

GENERATED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

# (名前, データ, encode の kwargs)。データは各パターンの容量に収まる範囲にしてある。
CASES: list[tuple[str, str, dict]] = [
    # プロファイル x ECC (compact / mono)
    ("compact_L", "MUTSUME COMPACT", dict(ecc="L")),
    ("compact_M", "MUTSUME COMPACT", dict(ecc="M")),
    ("compact_Q", "MUTSUME COMPACT", dict(ecc="Q")),
    ("compact_H", "MUTSUME COMPACT", dict(ecc="H")),
    # robust (射影変換対応プロファイル)
    ("robust_M", "https://example.com/robust", dict(profile="robust", ecc="M")),
    ("robust_Q", "https://example.com/robust", dict(profile="robust", ecc="Q")),
    # micro (最小フォーマット。mono 固定・ECC 固定)
    ("micro_numeric", "123456789", dict(profile="micro")),
    ("micro_alnum", "HELLO", dict(profile="micro")),
    ("micro_byte", "Hi!", dict(profile="micro")),
    # nano (中央ブルズアイ 1 個・最小規格。相似変換 = 正対前提)
    ("nano_min", "12", dict(profile="nano")),
    ("nano_numeric", "12345", dict(profile="nano")),
    ("nano_alnum", "HELLO", dict(profile="nano")),
    ("nano_byte", "Hi!", dict(profile="nano")),
    ("nano_max", "ABCDEFGH", dict(profile="nano")),
    # 文字モード (数字 / 英数 / バイト / 日本語)
    ("mode_numeric", "0123456789" * 4, dict(ecc="M")),
    ("mode_alnum", "ABCDEFGHIJ 0123456789 $%*+-./:", dict(ecc="M")),
    ("mode_byte", "mixed Text 123 !?#", dict(ecc="M")),
    ("mode_japanese", "六角形の二次元コード", dict(ecc="M")),
    # 混在 (セグメント分割が効くケース)
    ("mixed_segments", "ORDER-1234567890-ABCDE", dict(ecc="Q")),
]


class TestRoundtripImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(GENERATED, exist_ok=True)

    def _check(self, name: str, text: str, kwargs: dict,
               grid_lines: bool = True) -> None:
        sym = encode(text, **kwargs)
        png = os.path.join(GENERATED, f"{name}.png")
        svg = os.path.join(GENERATED, f"{name}.svg")
        sym.save(png, grid_lines=grid_lines)
        sym.save(svg, grid_lines=grid_lines)
        self.assertTrue(os.path.getsize(png) > 0, "PNG が空")
        self.assertTrue(os.path.getsize(svg) > 0, "SVG が空")

        res = decode(png)
        self.assertEqual(res.text, text, f"{name}: 復号テキスト不一致")
        # encode で指定したプロファイル / パレットが自動判別で戻ること
        self.assertEqual(res.profile, sym.profile, f"{name}: profile 不一致")
        self.assertEqual(res.palette, sym.palette, f"{name}: palette 不一致")
        self.assertIsNotNone(res.geometry, f"{name}: geometry が無い")

    def test_roundtrip_all_patterns(self) -> None:
        for name, text, kwargs in CASES:
            with self.subTest(pattern=name):
                self._check(name, text, kwargs)

    def test_roundtrip_no_grid_lines(self) -> None:
        """白セルの黒枠線を消したパターン (grid_lines=False) でも往復できること。

        黒セル同士を分ける白セパレータだけが残り、白セルは背景に溶ける。
        検出は収縮 + 連結成分で見ているので、枠線の有無に依存しない。
        """
        cases = [
            ("nogrid_compact", "NO GRID LINES", dict(ecc="Q")),
            ("nogrid_robust", "https://example.com/nogrid",
             dict(profile="robust", ecc="Q")),
            ("nogrid_micro", "123456789", dict(profile="micro")),
        ]
        for name, text, kwargs in cases:
            with self.subTest(pattern=name):
                self._check(name, text, kwargs, grid_lines=False)

    def test_decode_all_multiple_symbols(self) -> None:
        """2 つのコードを 1 枚に並べて、両方読めることを確認する。"""
        from PIL import Image

        a = encode("FIRST-CODE", ecc="M").to_image(cell_size=16)
        b = encode("SECOND-CODE", ecc="M").to_image(cell_size=16)
        gap = 40
        canvas = Image.new(
            "RGB",
            (a.width + b.width + gap * 3, max(a.height, b.height) + gap * 2),
            (255, 255, 255),
        )
        canvas.paste(a, (gap, gap))
        canvas.paste(b, (a.width + gap * 2, gap))
        path = os.path.join(GENERATED, "decode_all_two.png")
        canvas.save(path)

        results = decode_all(path, max_symbols=2)
        texts = sorted(r.text for r in results)
        self.assertEqual(texts, ["FIRST-CODE", "SECOND-CODE"])

    def test_decode_raises_on_blank(self) -> None:
        """コードの無い画像は MutsumeError (decode) / 空リスト (decode_all)。"""
        from PIL import Image

        blank = Image.new("RGB", (200, 200), (255, 255, 255))
        path = os.path.join(GENERATED, "blank.png")
        blank.save(path)
        with self.assertRaises(MutsumeError):
            decode(path)
        self.assertEqual(decode_all(path), [])


if __name__ == "__main__":
    unittest.main()

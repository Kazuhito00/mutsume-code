"""mutsume-code の単体テスト (標準ライブラリの unittest のみ使用)。

    venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from mutsume import MutsumeError, decode, encode  # noqa: E402
from mutsume.codec import (  # noqa: E402
    ECC_LEVELS,
    decode_grid_any_orientation,
    orient_grid,
    parity_count,
    payload_capacity,
)
from mutsume.detect import decode_image, decode_image_all, locate_image  # noqa: E402
from mutsume.layout import (  # noqa: E402
    DIRS,
    MAX_RADIUS,
    MIN_RADIUS,
    cell_count,
    get_layout,
    hex_distance,
    hex_ring,
    hex_spiral,
    mask_bit,
    rot60_n,
)
from mutsume.palette import PALETTE_NAMES  # noqa: E402
from mutsume.render import (  # noqa: E402
    draw_geometry,
    render_png,
    render_svg,
    render_text,
)
from mutsume.rs import RSDecodeError, rs_decode, rs_encode  # noqa: E402


class TestReedSolomon(unittest.TestCase):
    def test_roundtrip_no_error(self):
        data = bytes(range(32))
        cw = rs_encode(data, 10)
        self.assertEqual(len(cw), 42)
        fixed, n = rs_decode(cw, 10)
        self.assertEqual(fixed[:32], data)
        self.assertEqual(n, 0)

    def test_corrects_up_to_t_errors(self):
        rng = random.Random(20260811)
        for _ in range(300):
            k = rng.randint(1, 60)
            nsym = rng.choice([4, 8, 10, 16, 24])
            data = bytes(rng.randrange(256) for _ in range(k))
            cw = bytearray(rs_encode(data, nsym))
            t = rng.randint(0, nsym // 2)
            for p in rng.sample(range(len(cw)), t):
                cw[p] ^= rng.randrange(1, 256)
            fixed, n = rs_decode(bytes(cw), nsym)
            self.assertEqual(fixed[:k], data)
            self.assertLessEqual(n, t)

    def test_corrects_errors_and_erasures(self):
        """2e + f <= nsym の範囲で誤りと消失を同時に訂正できること。"""
        rng = random.Random(99)
        for _ in range(200):
            k = rng.randint(1, 60)
            nsym = rng.choice([8, 16, 24, 32])
            data = bytes(rng.randrange(256) for _ in range(k))
            cw = bytearray(rs_encode(data, nsym))
            f = rng.randint(0, min(nsym, len(cw)))
            e = rng.randint(0, (nsym - f) // 2)
            pos = rng.sample(range(len(cw)), f + e)
            for p in pos:
                cw[p] ^= rng.randrange(1, 256)
            fixed, _ = rs_decode(bytes(cw), nsym, erase_pos=pos[:f])
            self.assertEqual(fixed[:k], data)

    def test_erasures_double_the_capability(self):
        """消失位置が既知なら nsym 個まで訂正できる (誤りのみなら nsym/2)。"""
        rng = random.Random(5)
        k, nsym = 40, 16
        data = bytes(rng.randrange(256) for _ in range(k))
        cw = bytearray(rs_encode(data, nsym))
        pos = rng.sample(range(len(cw)), nsym)  # nsym 個 = 誤りのみでは訂正不能
        for p in pos:
            cw[p] ^= rng.randrange(1, 256)
        with self.assertRaises(RSDecodeError):
            rs_decode(bytes(cw), nsym)
        fixed, n = rs_decode(bytes(cw), nsym, erase_pos=pos)
        self.assertEqual(fixed[:k], data)
        self.assertEqual(n, nsym)

    def test_detects_excess_errors(self):
        rng = random.Random(7)
        data = bytes(rng.randrange(256) for _ in range(20))
        nsym = 8
        cw = bytearray(rs_encode(data, nsym))
        for p in range(nsym // 2 + 3):  # t を超える誤り
            cw[p] ^= 0xFF
        with self.assertRaises((RSDecodeError, AssertionError)):
            fixed, _ = rs_decode(bytes(cw), nsym)
            self.assertEqual(fixed[:20], data)  # 万一通ったら不一致で失敗させる


class TestLayout(unittest.TestCase):
    def test_ring_and_spiral(self):
        for k in range(0, 8):
            ring = hex_ring(k)
            self.assertEqual(len(ring), 6 * k if k else 1)
            self.assertEqual(len(set(ring)), len(ring))
            for c in ring:
                self.assertEqual(hex_distance(c), k)
        for r in range(MIN_RADIUS, 12):
            cells = hex_spiral(r)
            self.assertEqual(len(cells), cell_count(r))
            self.assertEqual(len(set(cells)), len(cells))

    def test_ring_corner_indices(self):
        k = 9
        ring = hex_ring(k)
        for i in range(6):
            self.assertEqual(ring[i * k], (DIRS[i][0] * k, DIRS[i][1] * k))

    def test_rotation_preserves_symbol(self):
        cells = set(hex_spiral(9))
        for n in range(6):
            self.assertEqual({rot60_n(c, n) for c in cells}, cells)

    def test_function_cells_disjoint(self):
        for r in range(MIN_RADIUS, MAX_RADIUS + 1):
            lay = get_layout(r)
            fmt = [c for g in lay.format_positions for c in g]
            self.assertEqual(len(fmt), 18)  # 6 ビット x 3 コピー
            self.assertEqual(len(set(fmt)), 18)
            self.assertEqual(len(lay.locator_cells), 21)
            self.assertEqual(len(lay.signature_cells), 6)
            reserved = set(lay.function_values) | set(fmt)
            self.assertEqual(len(reserved), 21 + 6 + 18)
            self.assertTrue(reserved.isdisjoint(lay.data_cells))
            self.assertEqual(len(lay.cells), len(reserved) + len(lay.data_cells))

    def test_masks_are_distinct(self):
        cells = hex_spiral(8)
        patterns = {m: tuple(mask_bit(m, c) for c in cells) for m in range(4)}
        self.assertEqual(len(set(patterns.values())), 4)
        for m, pat in patterns.items():
            ratio = sum(pat) / len(pat)
            self.assertTrue(0.2 < ratio < 0.8, f"mask {m} is unbalanced: {ratio}")


class TestCodecLogical(unittest.TestCase):
    def test_roundtrip_all_ecc(self):
        for ecc in ECC_LEVELS:
            with self.subTest(ecc=ecc):
                msg = f"mutsume-code / ECC={ecc} / 日本語テキスト".encode()
                sym = encode(msg, ecc=ecc)
                res = decode(sym)
                self.assertEqual(res.payload, msg)
                self.assertEqual(res.ecc, ecc)
                self.assertEqual(res.errors_corrected, 0)

    def test_empty_and_binary_payload(self):
        for payload in (b"", b"\x00", bytes(range(40))):
            with self.subTest(n=len(payload)):
                self.assertEqual(decode(encode(payload)).payload, payload)

    def test_capacity_boundary(self):
        for r in (8, 11, 14, 22, 30):
            for ecc in ECC_LEVELS:
                cap = payload_capacity(r, ecc)
                with self.subTest(r=r, ecc=ecc):
                    sym = encode(b"\xff" * cap, ecc=ecc, radius=r)
                    self.assertEqual(decode(sym).payload, b"\xff" * cap)
                    with self.assertRaises(MutsumeError):
                        encode(b"\xff" * (cap + 1), ecc=ecc, radius=r)

    def test_capacity_boundary_char_modes(self):
        for r in (10, 16, 24):
            for mode, ch in (("numeric", b"7"), ("alnum", b"Q")):
                cap = payload_capacity(r, "M", mode)
                with self.subTest(r=r, mode=mode):
                    sym = encode(ch * cap, ecc="M", radius=r)
                    self.assertEqual(decode(sym).segments, [(mode, cap)])
                    with self.assertRaises(MutsumeError):
                        encode(ch * (cap + 1), ecc="M", radius=r)

    def test_char_modes_beat_byte_mode(self):
        """数字/英数モードがバイトモードより小さいシンボルに収まること。"""
        digits = b"1234567890" * 6
        alnum = b"ABCDEFGHIJ" * 6
        raw = bytes(range(60))
        r_num = encode(digits).radius
        r_aln = encode(alnum).radius
        r_byte = encode(raw).radius
        self.assertLess(r_num, r_aln)
        self.assertLess(r_aln, r_byte)
        for data in (digits, alnum, raw):
            self.assertEqual(decode(encode(data)).payload, data)

    def test_mixed_segments_roundtrip(self):
        payload = "注文ID:ABC-12345 数量 000420 https://ex.jp".encode()
        sym = encode(payload, ecc="M")
        res = decode(sym)
        self.assertEqual(res.payload, payload)
        self.assertGreater(len(res.segments), 0)

    def test_auto_radius_grows(self):
        prev = 0
        for n in (1, 20, 60, 120, 174):  # 174 = ECC=M / R=25 の上限
            sym = encode(b"z" * n)
            self.assertGreaterEqual(sym.radius, prev)
            self.assertEqual(decode(sym).payload, b"z" * n)
            prev = sym.radius

    def test_all_orientations(self):
        msg = b"orientation test"
        sym = encode(msg, ecc="M")
        for mirrored in (False, True):
            for rot in range(6):
                with self.subTest(rot=rot, mirrored=mirrored):
                    observed = orient_grid(sym.grid, rot, mirrored)
                    res = decode_grid_any_orientation(observed, sym.radius)
                    self.assertEqual(res.payload, msg)

    def test_bit_errors_within_ecc(self):
        rng = random.Random(3)
        msg = b"error correction check"
        sym = encode(msg, ecc="H")
        lay = sym.layout
        nsym = parity_count(lay.data_byte_capacity, "H")
        t = nsym // 2
        # 1 バイト = 連続する 8 セル。t バイト分だけ壊す。
        grid = dict(sym.grid)
        for i in rng.sample(range(lay.data_byte_capacity), t):
            for cell in lay.data_cells[i * 8:(i + 1) * 8]:
                grid[cell] ^= 1
        res = decode_grid_any_orientation(grid, sym.radius)
        self.assertEqual(res.payload, msg)
        self.assertGreater(res.errors_corrected, 0)

    def test_mask_selection_is_recorded(self):
        for m in range(4):
            sym = encode(b"mask", mask=m)
            self.assertEqual(sym.mask_id, m)
            self.assertEqual(decode(sym).mask_id, m)


class TestBlocks(unittest.TestCase):
    def test_block_plan_is_consistent(self):
        from mutsume.codec import block_plan, interleave_order

        for r in range(MIN_RADIUS, MAX_RADIUS + 1):
            total = get_layout(r).data_byte_capacity
            for ecc in ECC_LEVELS:
                nsym = parity_count(total, ecc)
                plan = block_plan(total, nsym)
                with self.subTest(r=r, ecc=ecc):
                    self.assertEqual(sum(k + s for k, s in plan), total)
                    self.assertEqual(sum(s for _, s in plan), nsym)
                    self.assertTrue(all(k + s <= 255 for k, s in plan))
                    order = interleave_order(plan)
                    self.assertEqual(len(order), total)
                    self.assertEqual(len(set(order)), total)

    def test_large_symbol_uses_multiple_blocks(self):
        from mutsume.codec import block_plan

        total = get_layout(MAX_RADIUS).data_byte_capacity
        plan = block_plan(total, parity_count(total, "M"))
        self.assertGreater(len(plan), 1)
        payload = bytes(range(256)) * 1
        sym = encode(payload, ecc="M")
        self.assertEqual(decode(sym).payload, payload)

    def test_interleaving_spreads_a_burst(self):
        """連続バイトの破壊が全ブロックに分散すること。"""
        from mutsume.codec import block_plan, interleave_order

        plan = block_plan(600, 90)
        order = interleave_order(plan)
        hit = {b for b, _ in order[100:130]}
        self.assertEqual(len(hit), len(plan))


class TestRender(unittest.TestCase):
    def test_png_and_svg_and_text(self):
        sym = encode("render", ecc="M")
        img = render_png(sym, None, cell_size=10)
        self.assertGreater(img.width, 100)
        svg = render_svg(sym, None)
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count("<polygon"), len(sym.layout.cells))
        text = render_text(sym)
        self.assertEqual(len(text.splitlines()), 2 * sym.radius + 1)

    def test_dark_separator_is_drawn(self):
        """隣接する黒セルの間に白画素が入ること。"""
        import numpy as np

        from mutsume.layout import DIRS, to_cartesian
        from mutsume.render import _geometry

        sym = encode("separator check", ecc="M")
        cell_size = 20.0
        ox, oy, _, _ = _geometry(sym, cell_size, 1.5)

        pair = next(((c, n) for c in sym.layout.cells if sym.grid[c]
                     for n in [(c[0] + d[0], c[1] + d[1]) for d in DIRS]
                     if sym.grid.get(n)), None)
        self.assertIsNotNone(pair, "隣接する黒セルの組が見つからない")
        a, b = pair
        ax, ay = to_cartesian(a, cell_size)
        bx, by = to_cartesian(b, cell_size)
        mid = (int(round((ax + bx) / 2 + ox)), int(round((ay + by) / 2 + oy)))

        with_sep = np.asarray(render_png(sym, None, cell_size=cell_size).convert("L"))
        without = np.asarray(render_png(sym, None, cell_size=cell_size,
                                        dark_separator=False).convert("L"))
        # 共有辺の中点まわりの最大輝度で比べる (線幅 1px 程度なので 1 点だと外しうる)
        y, x = mid[1], mid[0]
        a = int(with_sep[y - 1:y + 2, x - 1:x + 2].max())
        b = int(without[y - 1:y + 2, x - 1:x + 2].max())
        self.assertLess(b, 90, "セパレータ無効時に境界が黒のままでない")
        self.assertGreater(a - b, 80, f"境界が白くなっていない ({b} -> {a})")

    def test_locators_are_detected_with_separators(self):
        """白セパレータが花形の内部にも入った状態で、3 つのロケータを検出できること。"""
        import numpy as np

        from mutsume.detect import find_locator_candidates, otsu_threshold
        from mutsume.layout import to_cartesian
        from mutsume.render import _geometry

        for text, ecc, size in (("locator", "M", 20.0), ("locator", "H", 12.0)):
            with self.subTest(ecc=ecc, size=size):
                sym = encode(text, ecc=ecc)
                img = render_png(sym, None, cell_size=size)
                gray = np.asarray(img.convert("L"))
                cands = find_locator_candidates(gray < otsu_threshold(gray))
                self.assertGreaterEqual(len(cands), 3)

                ox, oy, _, _ = _geometry(sym, size, 1.5)
                for lc in sym.layout.locator_centers:
                    x, y = to_cartesian(lc, size)
                    px, py = x + ox, y + oy
                    near = min(math.dist((px, py), (c.cx, c.cy)) for c in cands)
                    self.assertLess(near, size * 0.5,
                                    f"ロケータ {lc} が候補に上がっていない")


class TestImageDecode(unittest.TestCase):
    MSG = "https://example.com/mutsume-code?id=42"

    @classmethod
    def setUpClass(cls):
        cls.sym = encode(cls.MSG, ecc="Q")
        cls.img = render_png(cls.sym, None, cell_size=18)

    def _check(self, img):
        res = decode_image(img)
        self.assertEqual(res.payload, self.MSG.encode())
        return res

    def test_baseline(self):
        res = self._check(self.img)
        self.assertEqual(res.radius, self.sym.radius)
        self.assertEqual(res.errors_corrected, 0)

    def test_rotations(self):
        for ang in (7, 33, 90, 137, 180, 274):
            with self.subTest(angle=ang):
                self._check(self.img.rotate(ang, resample=Image.BICUBIC,
                                            expand=True, fillcolor=(247, 245, 240)))

    def test_mirrored(self):
        self._check(self.img.transpose(Image.FLIP_LEFT_RIGHT))

    def test_downscale(self):
        for s in (0.6, 0.4, 0.25):
            with self.subTest(scale=s):
                self._check(self.img.resize(
                    (int(self.img.width * s), int(self.img.height * s)), Image.LANCZOS))

    def test_blur_and_jpeg(self):
        self._check(self.img.filter(ImageFilter.GaussianBlur(2.0)))
        buf = io.BytesIO()
        self.img.save(buf, "JPEG", quality=35)
        buf.seek(0)
        self._check(Image.open(buf))

    def test_shear(self):
        sheared = self.img.transform(self.img.size, Image.AFFINE,
                                     (1, 0.2, -60, 0.05, 1, 0),
                                     resample=Image.BICUBIC,
                                     fillcolor=(247, 245, 240))
        self._check(sheared)

    def test_large_occlusion_uses_erasures(self):
        """消失訂正により、誤り訂正だけでは無理な広い遮蔽も読めること。"""
        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=18)
        d = ImageDraw.Draw(img)
        r = int(math.sqrt(0.15) * img.width * 0.5)
        cx, cy = img.width // 2, img.height // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 30, 30))
        res = decode_image(img)
        self.assertEqual(res.payload, self.MSG.encode())
        self.assertGreater(res.erasures_used, 0, "消失訂正が使われていない")

    def test_illumination_gradient_needs_adaptive(self):
        """強い照明ムラでは大域 Otsu が失敗し、適応的二値化で救われること。"""
        import numpy as np

        from mutsume.detect import (DetectionReport, _decode_binarized, _load_gray,
                                     binarize_otsu)

        arr = np.asarray(self.img).astype(np.float64)
        h, w, _ = arr.shape
        _, xx = np.mgrid[0:h, 0:w]
        shaded = np.clip(arr * (0.10 + 0.90 * (xx / w))[..., None], 0, 255)
        img = Image.fromarray(shaded.astype(np.uint8))

        gray, _ = _load_gray(img)
        with self.assertRaises(MutsumeError):
            _decode_binarized(binarize_otsu(gray), None, DetectionReport(),
                              ["compact"])

        rep = DetectionReport()
        res = decode_image(img, report=rep)
        self.assertEqual(res.payload, self.MSG.encode())
        self.assertIn("sauvola", rep.binarization)

    def test_small_occlusion(self):
        img = self.img.copy()
        d = ImageDraw.Draw(img)
        r = int(math.sqrt(0.05) * img.width * 0.5)
        cx, cy = img.width // 2, img.height // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 30, 30))
        res = self._check(img)
        self.assertGreater(res.errors_corrected, 0)

    def test_without_separator(self):
        """白セパレータなし (黒がベタ塗り) の描画も読めること。"""
        plain = render_png(self.sym, None, cell_size=18, dark_separator=False)
        self.assertEqual(decode_image(plain).payload, self.MSG.encode())

    def test_without_grid_lines(self):
        """白セルの黒枠線を消しても (白セパレータは残す) 読めること。"""
        import numpy as np

        img = render_png(self.sym, None, cell_size=18, grid_lines=False)
        self.assertEqual(decode_image(img).payload, self.MSG.encode())

        # 枠線が実際に消えていること: 白セルだけの領域が背景と地続きになり、
        # 画像全体の暗い画素の割合が下がる
        with_lines = np.asarray(render_png(self.sym, None, cell_size=18).convert("L"))
        without = np.asarray(img.convert("L"))
        self.assertLess((without < 128).mean(), (with_lines < 128).mean())

    def test_grid_lines_off_survives_degradation(self):
        for profile in ("compact", "robust"):
            sym = encode(self.MSG, ecc="Q", profile=profile)
            img = render_png(sym, None, cell_size=18, grid_lines=False)
            for name, im in (
                ("plain", img),
                ("rotate", img.rotate(33, resample=Image.BICUBIC, expand=True,
                                      fillcolor=(247, 245, 240))),
                ("scale", img.resize((int(img.width * 0.3), int(img.height * 0.3)),
                                     Image.LANCZOS)),
                ("blur", img.filter(ImageFilter.GaussianBlur(2.0))),
            ):
                with self.subTest(profile=profile, case=name):
                    self.assertEqual(decode_image(im).payload, self.MSG.encode())

    def test_radius_hint(self):
        res = decode_image(self.img, radius_hint=self.sym.radius)
        self.assertEqual(res.payload, self.MSG.encode())

    def test_blank_image_raises(self):
        with self.assertRaises(MutsumeError):
            decode_image(Image.new("RGB", (200, 200), (255, 255, 255)))

    def test_file_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sym.png")
            render_png(self.sym, path, cell_size=14)
            self.assertEqual(decode_image(path).payload, self.MSG.encode())


def _find_coeffs(target, source):
    import numpy as np

    A = []
    for (x, y), (u, v) in zip(target, source):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(A, float), np.array(source, float).reshape(8))


def tilt_image(img, t, bg=(247, 245, 240)):
    """上辺を t の割合だけ狭める台形歪み (斜めから撮った写真の模擬)。"""
    w, h = img.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(w * t, 0), (w * (1 - t), 0), (w, h), (0, h)]
    return img.transform((w, h), Image.PERSPECTIVE, _find_coeffs(dst, src),
                         resample=Image.BICUBIC, fillcolor=bg)


class TestRobustProfile(unittest.TestCase):
    MSG = "https://example.com/mutsume-code?id=42"

    @classmethod
    def setUpClass(cls):
        cls.sym = encode(cls.MSG, ecc="Q", profile="robust")
        cls.img = render_png(cls.sym, None, cell_size=18)

    def test_layout_shape(self):
        lay = self.sym.layout
        self.assertEqual(lay.profile, "robust")
        self.assertEqual(len(lay.locator_cells), 19 * 3)
        self.assertEqual(len(lay.alignment_cells), 7 * 3)
        self.assertEqual(len(lay.signature_cells), 9)
        self.assertEqual(len(lay.marker_centers), 6)

    def test_logical_roundtrip(self):
        self.assertEqual(decode(self.sym).payload, self.MSG.encode())

    def test_image_roundtrip(self):
        res = decode_image(self.img)
        self.assertEqual(res.payload, self.MSG.encode())
        self.assertEqual(res.profile, "robust")

    def test_locators_identified_in_one_shot(self):
        """2 重リング判定だけで、真のロケータ 3 個に絞れること。"""
        import numpy as np

        from mutsume.detect import (binarize_otsu, find_locator_candidates)
        from mutsume.layout import to_cartesian
        from mutsume.render import _geometry

        gray = np.asarray(self.img.convert("L"))
        cands = find_locator_candidates(binarize_otsu(gray).dark)
        rings = [c for c in cands if c.is_double_ring]
        self.assertEqual(len(rings), 3, "2 重リング候補がちょうど 3 個でない")

        ox, oy, _, _ = _geometry(self.sym, 18.0, 1.5)
        for lc in self.sym.layout.locator_centers:
            x, y = to_cartesian(lc, 18.0)
            near = min(math.dist((x + ox, y + oy), (c.cx, c.cy)) for c in rings)
            self.assertLess(near, 9.0)

    def test_perspective_decode(self):
        """射影歪み (台形) を 6 点ホモグラフィで補正して読めること。"""
        for t in (0.08, 0.15):
            with self.subTest(tilt=t):
                res = decode_image(tilt_image(self.img, t))
                self.assertEqual(res.payload, self.MSG.encode())
                self.assertTrue(res.perspective, "ホモグラフィが使われていない")

    def test_compact_cannot_do_perspective(self):
        """compact はアフィンまでなので、同じ歪みでは読めない (仕様上の差)。"""
        compact = render_png(encode(self.MSG, ecc="Q", profile="compact"), None,
                             cell_size=18)
        with self.assertRaises(MutsumeError):
            decode_image(tilt_image(compact, 0.15), profile="compact")

    def test_rotation_and_scale(self):
        for ang in (33, 137):
            with self.subTest(angle=ang):
                rot = self.img.rotate(ang, resample=Image.BICUBIC, expand=True,
                                      fillcolor=(247, 245, 240))
                self.assertEqual(decode_image(rot).payload, self.MSG.encode())
        small = self.img.resize((int(self.img.width * 0.4), int(self.img.height * 0.4)),
                                Image.LANCZOS)
        self.assertEqual(decode_image(small).payload, self.MSG.encode())

    def test_capacity_cost_of_markers(self):
        """robust のマーカー増加分だけ容量が減り、大きいシンボルほど相対コストが下がる。"""
        for r in (10, 20, 40):
            c = payload_capacity(r, "M", "byte", "compact")
            b = payload_capacity(r, "M", "byte", "robust")
            self.assertLess(b, c)
        loss_small = 1 - payload_capacity(10, "M", "byte", "robust") / \
            payload_capacity(10, "M", "byte", "compact")
        loss_big = 1 - payload_capacity(40, "M", "byte", "robust") / \
            payload_capacity(40, "M", "byte", "compact")
        self.assertLess(loss_big, loss_small)


class TestGeometry(unittest.TestCase):
    """画像上の座標 (ファインダ位置・外形・ホモグラフィ) の取得。"""

    MSG = "https://example.com/mutsume-code?id=42"
    CELL = 18.0

    def _truth(self, sym, cell_size=None):
        """レンダリング時のセル中心座標 (答え合わせ用)。"""
        from mutsume.layout import to_cartesian
        from mutsume.render import _geometry

        size = cell_size or self.CELL
        ox, oy, _, _ = _geometry(sym, size, 1.5)
        return {c: (to_cartesian(c, size)[0] + ox, to_cartesian(c, size)[1] + oy)
                for c in sym.layout.cells}

    def test_geometry_attached_only_for_images(self):
        sym = encode(self.MSG, ecc="Q")
        self.assertIsNone(decode(sym).geometry)
        res = decode_image(render_png(sym, None, cell_size=self.CELL))
        self.assertIsNotNone(res.geometry)

    def test_finder_positions_match_rendering(self):
        for profile in ("compact", "robust", "micro"):
            data = "123456789" if profile == "micro" else self.MSG
            sym = encode(data, ecc="M" if profile == "micro" else "Q",
                         profile=profile)
            truth = self._truth(sym)
            g = decode_image(render_png(sym, None, cell_size=self.CELL)).geometry
            with self.subTest(profile=profile):
                self.assertEqual(len(g.finders), 3)
                for cell, got in zip(sym.layout.locator_centers, g.finders):
                    self.assertLess(math.dist(truth[cell], got), 2.0)
                self.assertEqual(len(g.alignments),
                                 3 if profile == "robust" else 0)
                for cell, got in zip(sym.layout.alignment_centers, g.alignments):
                    self.assertLess(math.dist(truth[cell], got), 2.0)

    def test_coordinates_are_in_original_image_space(self):
        """作業画像は長辺 1400px に縮小されるが、座標は元のスケールで返ること。"""
        from mutsume.detect import MAX_WORK_SIZE

        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=self.CELL)
        big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        self.assertGreater(max(big.size), MAX_WORK_SIZE, "縮小が起きる大きさが前提")

        truth = self._truth(sym)
        g = decode_image(big).geometry
        for cell, got in zip(sym.layout.locator_centers, g.finders):
            want = (truth[cell][0] * 2, truth[cell][1] * 2)
            self.assertLess(math.dist(want, got), 5.0)
        self.assertAlmostEqual(g.cell_size, self.CELL * math.sqrt(3) * 2, delta=1.0)

    def test_outline_and_bbox_cover_the_symbol(self):
        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=self.CELL)
        g = decode_image(img).geometry
        self.assertEqual(len(g.outline), 6)
        x0, y0, x1, y1 = g.bbox
        # 外形はすべてのセル中心を含み、画像内に収まる
        for p in g.cell_centers().values():
            self.assertTrue(x0 <= p[0] <= x1 and y0 <= p[1] <= y1)
        self.assertGreaterEqual(x0, -2)
        self.assertLessEqual(x1, img.width + 2)
        # クワイエットゾーンぶん、画像より内側にある
        self.assertGreater(x0, 5)

    def test_mirrored_flag(self):
        """鏡映で写ったシンボルを geometry.mirrored で検出できること。"""
        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=self.CELL)
        self.assertFalse(decode_image(img).geometry.mirrored)
        res = decode_image(img.transpose(Image.FLIP_LEFT_RIGHT))
        self.assertEqual(res.payload, self.MSG.encode())
        self.assertTrue(res.geometry.mirrored)

    def test_corner0_marks_orientation(self):
        """corner0 (向きの指標) が回転に追従すること。"""
        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=self.CELL)
        g0 = decode_image(img).geometry
        self.assertEqual(g0.corner0, g0.outline[0])
        # 90 度回転 (PIL は反時計回り、画像座標は y 下向き) -> corner0 も回る
        g90 = decode_image(img.rotate(90, expand=True,
                                      fillcolor=(247, 245, 240))).geometry
        v0 = (g0.corner0[0] - g0.center[0], g0.corner0[1] - g0.center[1])
        v90 = (g90.corner0[0] - g90.center[0], g90.corner0[1] - g90.center[1])
        ang = math.degrees(math.atan2(v90[1], v90[0]) - math.atan2(v0[1], v0[0]))
        self.assertLess(abs((ang + 90 + 180) % 360 - 180), 3.0)

    def test_rotation_is_reported(self):
        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=self.CELL)
        base = decode_image(img).geometry.rotation_deg
        for ang in (30, 90):
            rot = img.rotate(ang, resample=Image.BICUBIC, expand=True,
                             fillcolor=(247, 245, 240))
            got = decode_image(rot).geometry.rotation_deg
            # PIL の rotate は反時計回り、画像座標系は y 下向き
            diff = (got - base + ang + 180) % 360 - 180
            with self.subTest(angle=ang):
                self.assertLess(abs(diff), 3.0, f"{base} -> {got}")

    def test_cell_polygon_and_centers(self):
        sym = encode(self.MSG, ecc="Q")
        g = decode_image(render_png(sym, None, cell_size=self.CELL)).geometry
        poly = g.cell_polygon((0, 0))
        self.assertEqual(len(poly), 6)
        c = g.cell_center((0, 0))
        for p in poly:
            self.assertAlmostEqual(math.dist(c, p), self.CELL, delta=1.0)
        centers = g.cell_centers()
        self.assertEqual(len(centers), len(sym.layout.cells))

    def test_to_dict_is_json_serializable(self):
        import json

        sym = encode(self.MSG, ecc="Q")
        g = decode_image(render_png(sym, None, cell_size=self.CELL)).geometry
        text = json.dumps(g.to_dict())
        back = json.loads(text)
        self.assertEqual(len(back["finders"]), 3)
        self.assertEqual(len(back["homography"]), 3)

    def test_locate_image(self):
        sym = encode(self.MSG, ecc="Q")
        g = locate_image(render_png(sym, None, cell_size=self.CELL))
        self.assertEqual(g.radius, sym.radius)
        self.assertEqual(g.profile, "compact")

    def test_draw_geometry(self):
        sym = encode(self.MSG, ecc="Q")
        img = render_png(sym, None, cell_size=self.CELL)
        g = decode_image(img).geometry
        out = draw_geometry(img, g)
        self.assertEqual(out.size, img.size)
        self.assertNotEqual(list(out.getdata()), list(img.convert("RGB").getdata()))

    def test_perspective_geometry(self):
        sym = encode(self.MSG, ecc="Q", profile="robust")
        img = render_png(sym, None, cell_size=self.CELL)
        g = decode_image(tilt_image(img, 0.15)).geometry
        self.assertTrue(g.perspective)
        # 台形なので上辺が下辺より短い (外形コーナー 1-2 が上、4-5 が下)
        top = math.dist(g.outline[1], g.outline[2])
        bottom = math.dist(g.outline[4], g.outline[5])
        self.assertLess(top, bottom * 0.9)


class TestLabelComponents(unittest.TestCase):
    """連結成分ラベリング (numpy 実装) が素朴な実装と一致すること。"""

    @staticmethod
    def _reference(mask, min_run):
        """行ごとのループ + 総当たり隣接判定による素朴な実装。"""
        import numpy as np

        h, w = mask.shape
        parent = []

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        runs, prev = [], []
        for y in range(h):
            row = np.concatenate(([0], mask[y].astype(np.int8), [0]))
            d = np.diff(row)
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            if min_run > 1 and len(starts):
                keep = (ends - starts) >= min_run
                starts, ends = starts[keep], ends[keep]
            cur = []
            for xs, xe in zip(starts, ends):
                rid = len(parent)
                parent.append(rid)
                runs.append((y, int(xs), int(xe), rid))
                for pxs, pxe, pid in prev:
                    if pxs < xe and xs < pxe:
                        ra, rb = find(rid), find(pid)
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
                cur.append((int(xs), int(xe), rid))
            prev = cur

        acc = {}
        for y, xs, xe, rid in runs:
            root, n = find(rid), xe - xs
            a = acc.setdefault(root, [0, 0.0, 0, xs, xe - 1, y, y])
            a[0] += n
            a[1] += (xs + xe - 1) * n / 2.0
            a[2] += y * n
            a[3], a[4] = min(a[3], xs), max(a[4], xe - 1)
            a[5], a[6] = min(a[5], y), max(a[6], y)
        return sorted((int(v[0]), round(v[1] / v[0], 6), round(v[2] / v[0], 6),
                       v[3], v[4], v[5], v[6]) for v in acc.values())

    @staticmethod
    def _key(comps):
        return sorted((c.area, round(c.cx, 6), round(c.cy, 6),
                       c.x0, c.x1, c.y0, c.y1) for c in comps)

    def test_matches_naive_implementation(self):
        import numpy as np

        from mutsume.detect import MIN_RUN, label_components

        rng = np.random.default_rng(0)
        for i in range(30):
            h, w = int(rng.integers(5, 70)), int(rng.integers(5, 70))
            p = float(rng.uniform(0.05, 0.7))
            if i % 3:
                mask = rng.random((h, w)) < p
            else:  # セルに近いブロック状
                mask = np.kron(rng.random((max(1, h // 6), max(1, w // 6))) < p,
                               np.ones((6, 6), bool))[:h, :w]
            with self.subTest(i=i, shape=(h, w)):
                self.assertEqual(self._key(label_components(mask)),
                                 self._reference(mask, MIN_RUN))

    def test_edge_cases(self):
        import numpy as np

        from mutsume.detect import MIN_RUN, label_components

        cases = {
            "empty": np.zeros((10, 10), bool),
            "full": np.ones((10, 10), bool),
            "one_row": np.ones((1, 20), bool),
            "one_col": np.ones((20, 1), bool),
            "checker": np.indices((20, 20)).sum(0) % 2 == 0,
            "two_blobs": np.zeros((20, 20), bool),
        }
        cases["two_blobs"][2:8, 2:8] = True
        cases["two_blobs"][12:18, 12:18] = True
        for name, mask in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self._key(label_components(mask)),
                                 self._reference(mask, MIN_RUN))
        self.assertEqual(len(label_components(cases["two_blobs"])), 2)


class TestMultiSymbol(unittest.TestCase):
    """1 枚に複数のコードがある場合。"""

    def _canvas(self, specs, gap=60):
        """複数シンボルを横に並べた 1 枚を作る。"""
        imgs = [render_png(encode(t, ecc="Q", **kw), None, cell_size=cs)
                for t, kw, cs in specs]
        w = sum(i.width for i in imgs) + gap * (len(imgs) + 1)
        h = max(i.height for i in imgs) + gap * 2
        canvas = Image.new("RGB", (w, h), (247, 245, 240))
        x = gap
        for im in imgs:
            canvas.paste(im, (x, gap))
            x += im.width + gap
        return canvas

    def test_finds_every_symbol(self):
        specs = [("FIRST-CODE-1", {}, 14.0),
                 ("SECOND-CODE-2", {"profile": "robust"}, 14.0),
                 ("123456789", {"profile": "micro"}, 20.0)]
        canvas = self._canvas(specs)
        res = decode_image_all(canvas, max_symbols=5)
        got = {r.text for r in res}
        self.assertEqual(got, {t for t, _, _ in specs}, f"got {got}")

    def test_each_has_its_own_geometry(self):
        canvas = self._canvas([("LEFT-ONE", {}, 14.0), ("RIGHT-TWO", {}, 14.0)])
        res = decode_image_all(canvas, max_symbols=4)
        self.assertEqual(len(res), 2)
        centers = sorted(r.geometry.center[0] for r in res)
        self.assertGreater(centers[1] - centers[0], 100, "位置が分かれていない")
        for r in res:
            x0, y0, x1, y1 = r.geometry.bbox
            self.assertGreater(x1 - x0, 50)

    def test_single_symbol_returns_one(self):
        img = render_png(encode("ONLY-ONE", ecc="Q"), None, cell_size=14)
        res = decode_image_all(img, max_symbols=4)
        self.assertEqual([r.text for r in res], ["ONLY-ONE"])

    def test_no_symbol_returns_empty(self):
        self.assertEqual(
            decode_image_all(Image.new("RGB", (300, 300), (255, 255, 255))), [])

    def test_tracking_follows_small_motion(self):
        """前フレームの姿勢 (hints) で、少し動いた次のフレームを追従できること。"""
        sym = encode("TRACK-ME-42", ecc="Q")
        tile = render_png(sym, None, cell_size=10)
        canvas = Image.new("RGB", (tile.width + 80, tile.height + 80),
                           (247, 245, 240))

        def frame(dx, dy, angle=0.0):
            f = canvas.copy()
            t = tile.rotate(angle, resample=Image.BICUBIC, expand=False,
                            fillcolor=(247, 245, 240)) if angle else tile
            f.paste(t, (40 + dx, 40 + dy))
            return f

        first = decode_image_all(frame(0, 0), max_symbols=1)
        self.assertEqual(len(first), 1)
        hints = [first[0].geometry]

        # 平行移動 + わずかな回転をトラッキングで追う
        for dx, dy, ang in ((4, 2, 0), (8, 5, 2), (12, 3, 4)):
            res = decode_image_all(frame(dx, dy, ang), max_symbols=1,
                                   hints=hints, hints_only=True)
            self.assertEqual(len(res), 1, f"dx={dx} dy={dy} ang={ang}")
            self.assertEqual(res[0].payload, b"TRACK-ME-42")
            # 位置も移動に追従している
            g0, g1 = hints[0], res[0].geometry
            self.assertLess(abs((g1.center[0] - g0.center[0])), 25)
            hints = [g1]

    def test_tracking_gives_up_when_symbol_leaves(self):
        """シンボルが消えたら hints_only では空を返す (誤追従しない)。"""
        sym = encode("GONE-SOON", ecc="Q")
        img = render_png(sym, None, cell_size=10)
        first = decode_image_all(img, max_symbols=1)
        self.assertEqual(len(first), 1)
        blank = Image.new("RGB", img.size, (247, 245, 240))
        res = decode_image_all(blank, max_symbols=1,
                               hints=[first[0].geometry], hints_only=True)
        self.assertEqual(res, [])


class TestMicroProfile(unittest.TestCase):
    """Micro QR 相当の最小フォーマット。"""

    def test_layout_has_locators_only(self):
        from mutsume.layout import max_radius, min_radius

        self.assertEqual(min_radius("micro"), 5)
        self.assertEqual(max_radius("micro"), 8)
        for r in range(5, 9):
            lay = get_layout(r, "micro")
            with self.subTest(r=r):
                self.assertEqual(len(lay.locator_cells), 21)
                self.assertEqual(lay.signature_cells, ())
                self.assertEqual(lay.alignment_cells, ())
                self.assertEqual(lay.format_positions, ())
                self.assertEqual(len(lay.function_values), 21)
                self.assertEqual(len(lay.cells),
                                 len(lay.data_cells) + 21)

    def test_beats_compact_on_small_symbols(self):
        from mutsume.codec import micro_capacity

        # R=7 は compact だと数字 9 桁。micro なら大幅に増える
        self.assertGreater(micro_capacity(7, "numeric"),
                           payload_capacity(7, "M", "numeric") * 2)
        # compact が成立しない R=5, 6 でも入る
        self.assertGreater(micro_capacity(5, "numeric"), 0)
        self.assertGreater(micro_capacity(6, "numeric"), 0)

    def test_logical_roundtrip(self):
        for data in (b"1", b"1234567", b"123456789", b"0123456789" * 2,
                     b"ABCDE-123", b"hi", b"\x00\xff"):
            with self.subTest(data=data):
                sym = encode(data, profile="micro")
                res = decode(sym)
                self.assertEqual(res.payload, data)
                self.assertEqual(res.profile, "micro")
                self.assertEqual(res.mask_id, sym.mask_id)

    def test_capacity_boundary(self):
        from mutsume.codec import micro_capacity

        for r in range(5, 9):
            for mode, ch in (("numeric", b"7"), ("alnum", b"Q"), ("byte", b"\xff")):
                cap = micro_capacity(r, mode)
                with self.subTest(r=r, mode=mode):
                    sym = encode(ch * cap, profile="micro", radius=r)
                    self.assertEqual(decode(sym).payload, ch * cap)
                    with self.assertRaises(MutsumeError):
                        encode(ch * (cap + 1), profile="micro", radius=r)

    def test_image_roundtrip(self):
        data = "123456789"
        sym = encode(data, profile="micro")
        self.assertEqual(sym.radius, 6)
        for grid in (True, False):
            img = render_png(sym, None, cell_size=30, grid_lines=grid)
            for name, im in (
                ("plain", img),
                ("rotate", img.rotate(33, resample=Image.BICUBIC, expand=True,
                                      fillcolor=(247, 245, 240))),
                ("mirror", img.transpose(Image.FLIP_LEFT_RIGHT)),
                ("scale", img.resize((int(img.width * 0.3), int(img.height * 0.3)),
                                     Image.LANCZOS)),
                ("blur", img.filter(ImageFilter.GaussianBlur(2.0))),
            ):
                with self.subTest(grid_lines=grid, case=name):
                    res = decode_image(im)
                    self.assertEqual(res.payload, data.encode())
                    self.assertEqual(res.profile, "micro")

    def test_mask_and_radius_are_brute_forced(self):
        """フォーマット領域がなくても、マスク / 半径 / 向きを復元できること。"""
        data = "0123456789"
        for m in range(4):
            sym = encode(data, profile="micro", mask=m)
            with self.subTest(mask=m):
                img = render_png(sym, None, cell_size=30)
                res = decode_image(img)
                self.assertEqual(res.payload, data.encode())
                self.assertEqual(res.mask_id, m)

    def test_rejects_oversized_payload(self):
        with self.assertRaises(MutsumeError):
            encode(b"x" * 200, profile="micro")


class TestColorPalettes(unittest.TestCase):
    MSG = "https://example.com/mutsume-code?id=42&color=on"

    def test_bits_per_cell(self):
        from mutsume.palette import bits_per_cell

        self.assertEqual(bits_per_cell("mono"), 1)

    def test_mono_value_1_is_dark(self):
        """機能セルの規約 (1 = 暗い) とパレットの並びが一致していること。"""
        from mutsume.palette import rgb255

        self.assertEqual(rgb255("mono", 1), (0, 0, 0))
        self.assertEqual(rgb255("mono", 0), (255, 255, 255))

    def test_logical_roundtrip(self):
        for pal in PALETTE_NAMES:
            for ecc in ECC_LEVELS:
                with self.subTest(palette=pal, ecc=ecc):
                    sym = encode(self.MSG, ecc=ecc, palette=pal)
                    res = decode(sym)
                    self.assertEqual(res.payload, self.MSG.encode())
                    self.assertEqual(res.palette, pal)


class TestFuzzRoundtrip(unittest.TestCase):
    def test_random_payloads_through_png(self):
        rng = random.Random(1234)
        for i in range(6):
            n = rng.randint(0, 60)
            payload = bytes(rng.randrange(256) for _ in range(n))
            ecc = rng.choice(ECC_LEVELS)
            with self.subTest(i=i, n=n, ecc=ecc):
                sym = encode(payload, ecc=ecc)
                img = render_png(sym, None, cell_size=14)
                self.assertEqual(decode_image(img).payload, payload)


class TestCli(unittest.TestCase):
    def test_encode_decode_via_cli(self):
        import tempfile

        from mutsume.cli import main

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cli.png")
            self.assertEqual(main(["encode", "CLI TEST", "-o", path]), 0)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(main(["decode", path]), 0)
            self.assertEqual(main(["info"]), 0)


if __name__ == "__main__":
    unittest.main()

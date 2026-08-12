"""GF(256) 演算と Reed-Solomon 符号 (systematic)。

- 既約多項式: 0x11D (QR コードと同じ)
- 生成多項式の根: alpha^0 .. alpha^(nsym-1)
- 誤り訂正: Berlekamp-Massey + Chien search + Forney
"""

from __future__ import annotations

import numpy as np

PRIM = 0x11D
FIELD = 256

GF_EXP = [0] * 512
GF_LOG = [0] * 256


def _init_tables() -> None:
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM
    for i in range(255, 512):
        GF_EXP[i] = GF_EXP[i - 255]


_init_tables()

# numpy 版のテーブル (症候群計算の一括化用)
GF_EXP_NP = np.array(GF_EXP, dtype=np.uint8)
GF_LOG_NP = np.array(GF_LOG, dtype=np.int64)

# この問題サイズ (要素数の積) を下回ったら numpy ではなくスカラーで計算する。
# 小さい配列では numpy の呼び出しコストが本体を上回るため。
NUMPY_CROSSOVER = 512


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] + 255 - GF_LOG[b]) % 255]


def gf_inv(a: int) -> int:
    return gf_div(1, a)


def gf_pow(a: int, n: int) -> int:
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] * n) % 255]


# --- 多項式ユーティリティ (係数は低次から順に並べる) ---------------------------


def poly_add(p: list[int], q: list[int]) -> list[int]:
    n = max(len(p), len(q))
    out = [0] * n
    for i, v in enumerate(p):
        out[i] ^= v
    for i, v in enumerate(q):
        out[i] ^= v
    return out


def poly_mul(p: list[int], q: list[int]) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi == 0:
            continue
        for j, qj in enumerate(q):
            if qj:
                out[i + j] ^= gf_mul(pi, qj)
    return out


def poly_scale(p: list[int], s: int) -> list[int]:
    return [gf_mul(c, s) for c in p]


def poly_eval_low_first(p: list[int], x: int) -> int:
    """p[0] + p[1]x + ... を x で評価。"""
    acc = 0
    for c in reversed(p):
        acc = gf_mul(acc, x) ^ c
    return acc


def poly_deriv(p: list[int]) -> list[int]:
    """標数 2 における形式微分 (偶数次の項は消える)。"""
    out = [0] * max(1, len(p) - 1)
    for i in range(1, len(p)):
        if i % 2 == 1:
            out[i - 1] = p[i]
    return out


# --- RS 符号化 ----------------------------------------------------------------


def rs_generator_poly(nsym: int) -> list[int]:
    g = [1]
    for i in range(nsym):
        g = poly_mul(g, [GF_EXP[i], 1])  # (alpha^i + x)
    return g


def rs_encode(data: bytes | list[int], nsym: int) -> bytes:
    """systematic RS。戻り値 = data + parity(nsym バイト)。"""
    data = list(data)
    if len(data) + nsym > 255:
        raise ValueError(f"RS codeword too long: {len(data)}+{nsym} > 255")
    gen = rs_generator_poly(nsym)  # 低次から / 最高次係数は 1
    # 除算 (係数配列は高次から扱った方が素直なので反転して処理する)
    gen_hi = list(reversed(gen))  # [1, g_{n-1}, ..., g_0]
    out = data + [0] * nsym
    for i in range(len(data)):
        coef = out[i]
        if coef:
            for j in range(1, len(gen_hi)):
                out[i + j] ^= gf_mul(gen_hi[j], coef)
    return bytes(data) + bytes(out[len(data):])


# --- RS 復号 ------------------------------------------------------------------


class RSDecodeError(Exception):
    pass


def _syndromes(r: list[int], nsym: int) -> list[int]:
    """S_j = R(alpha^j) (j = 0..nsym-1)。r は高次から並んだ受信語。

    S_j = XOR_i gf_mul(r_i, alpha^(j * (n-1-i)))。gf_mul は exp/log テーブルで
    表せるので、非ゼロ係数だけを numpy で一括評価する。Horner の二重ループだと
    n * nsym 回の gf_mul (実写ベンチで 1 フレーム 26 万回) が全部 Python になる。
    """
    n = len(r)
    if n * nsym < NUMPY_CROSSOVER:
        synd = []
        for j in range(nsym):
            a = GF_EXP[j % 255]
            acc = 0
            for c in r:  # Horner (高次から)
                acc = gf_mul(acc, a) ^ c
            synd.append(acc)
        return synd

    arr = np.asarray(r, dtype=np.uint8)
    nz = np.flatnonzero(arr)
    if nz.size == 0:
        return [0] * nsym
    logs = GF_LOG_NP[arr[nz]]                      # (m,)
    pw = ((n - 1 - nz) % 255).astype(np.int64)     # (m,) 各係数の指数
    js = np.arange(nsym, dtype=np.int64)[:, None]  # (nsym, 1)
    e = logs[None, :] + (js * pw[None, :]) % 255   # 各項 < 510 で GF_EXP に収まる
    vals = GF_EXP_NP[e]                            # (nsym, m)
    return [int(v) for v in np.bitwise_xor.reduce(vals, axis=1)]


def _berlekamp_massey(synd: list[int], nsym: int) -> list[int]:
    """誤り位置多項式 Lambda(x) を返す (低次から)。"""
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for n in range(nsym):
        d = synd[n]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= gf_mul(C[i], synd[n - i])
        if d == 0:
            m += 1
            continue
        coef = gf_div(d, b)
        shifted = [0] * m + poly_scale(B, coef)
        if 2 * L <= n:
            T = C[:]
            C = poly_add(C, shifted)
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            C = poly_add(C, shifted)
            m += 1
    while len(C) > 1 and C[-1] == 0:
        C.pop()
    return C


def _chien_search(lam: list[int], n: int) -> list[int]:
    """Lambda(alpha^-p) = 0 となる指数 p を列挙 (p は多項式次数の位置)。

    n 点の評価を Horner のまま numpy へ持ち上げる (係数ごとに全点を一括更新)。
    失敗する復号試行が多い実写では、ここが Python ループだと積み上がる。
    """
    if n * len(lam) < NUMPY_CROSSOVER:
        roots = []
        for p in range(n):
            x = GF_EXP[(255 - (p % 255)) % 255]  # alpha^-p
            if poly_eval_low_first(lam, x) == 0:
                roots.append(p)
        return roots

    logx = (255 - (np.arange(n, dtype=np.int64) % 255)) % 255  # log(alpha^-p)
    acc = np.zeros(n, dtype=np.int64)
    for c in reversed(lam):
        nz = acc != 0
        acc[nz] = GF_EXP_NP[GF_LOG_NP[acc[nz]] + logx[nz]]
        acc ^= c
    return np.flatnonzero(acc == 0).tolist()


def _errata_locator(exponents: list[int]) -> list[int]:
    """既知位置 (指数) から消失位置多項式 prod(1 + alpha^p x) を作る。"""
    loc = [1]
    for p in exponents:
        loc = poly_mul(loc, [1, GF_EXP[p % 255]])
    return loc


def _forney_syndromes(synd: list[int], exponents: list[int]) -> list[int]:
    """消失位置を除去した修正症候群。残りの誤りは通常の BM で解ける。"""
    fsynd = list(synd)
    for p in exponents:
        x = GF_EXP[p % 255]
        for j in range(len(fsynd) - 1):
            fsynd[j] = gf_mul(fsynd[j], x) ^ fsynd[j + 1]
    return fsynd


def rs_decode(codeword: bytes | list[int], nsym: int,
              erase_pos: list[int] | None = None) -> tuple[bytes, int]:
    """誤り訂正して (訂正後の全符号語, 訂正したシンボル数) を返す。

    erase_pos に「壊れていると分かっている位置 (符号語のリスト添字)」を渡すと
    消失訂正になる。誤り e 個 + 消失 f 個は 2e + f <= nsym まで訂正できる。
    訂正不能なら RSDecodeError。
    """
    r = list(codeword)
    n = len(r)
    if n > 255:
        raise ValueError("codeword too long")

    erase_pos = sorted(set(erase_pos or ()))
    if any(not (0 <= p < n) for p in erase_pos):
        raise ValueError("erasure position out of range")
    if len(erase_pos) > nsym:
        raise RSDecodeError(f"too many erasures ({len(erase_pos)} > {nsym})")

    synd = _syndromes(r, nsym)
    if max(synd) == 0:
        return bytes(r), 0

    # 受信語 index k <-> 多項式の指数 p = n-1-k
    erase_exp = [n - 1 - k for k in erase_pos]

    if erase_exp:
        e_loc = _errata_locator(erase_exp)
        fsynd = _forney_syndromes(synd, erase_exp)
        err_loc = _berlekamp_massey(fsynd, nsym - len(erase_exp))
        lam = poly_mul(err_loc, e_loc)
        n_extra = len(err_loc) - 1
    else:
        lam = _berlekamp_massey(synd, nsym)
        n_extra = len(lam) - 1

    while len(lam) > 1 and lam[-1] == 0:
        lam.pop()
    nerr = len(lam) - 1
    if nerr == 0:
        raise RSDecodeError("syndromes non-zero but no error locator found")
    if 2 * n_extra + len(erase_exp) > nsym:
        raise RSDecodeError(
            f"beyond correction capability (2*{n_extra} + {len(erase_exp)} > {nsym})")

    roots = _chien_search(lam, n)
    if len(roots) != nerr:
        raise RSDecodeError("error locator has wrong number of roots")

    # Omega(x) = S(x) * Lambda(x) mod x^nsym
    omega = poly_mul(synd, lam)[:nsym]
    lam_d = poly_deriv(lam)

    corrected = r[:]
    for p in roots:
        xi = GF_EXP[p % 255]
        xi_inv = GF_EXP[(255 - (p % 255)) % 255]
        num = poly_eval_low_first(omega, xi_inv)
        den = poly_eval_low_first(lam_d, xi_inv)
        if den == 0:
            raise RSDecodeError("Forney denominator is zero")
        mag = gf_mul(xi, gf_div(num, den))
        k = n - 1 - p
        if not (0 <= k < n):
            raise RSDecodeError("error position out of range")
        corrected[k] ^= mag

    if max(_syndromes(corrected, nsym)) != 0:
        raise RSDecodeError("correction failed (syndromes non-zero)")
    return bytes(corrected), len(roots)

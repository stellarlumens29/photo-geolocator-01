#!/usr/bin/env python3
"""Code-only photo geolocator for make.fun photo-geolocator-01.

No model calls, no network beyond the trial env, stdlib only.
Strategy: decode each JPEG (baseline, written here from the JPEG
spec), measure colour / texture / road-marking signals at full
resolution, and place the photo on a small set of region anchors
with a conservative prior. Answers are a posterior-weighted
geographic mean, so an uncertain photo lands near the
prior-heavy northern-Eurasia region instead of jumping to a
faraway guess. Deterministic: everything derives from the photo
pixels and MAKEFUN_TRIAL_SEED only.
"""

VERSION = "1.1.0"

import hashlib
import json
import math
import os
import random
import sys
import urllib.request

# ---------------------------------------------------------------- JPEG ----
# Minimal baseline-JPEG decoder (SOF0, Huffman, 8x8 DCT).
# Written for this entry from the JPEG specification (ITU T.81):
# markers, quantisation tables, canonical Huffman codes, zig-zag
# order, DC difference coding, dequantisation and the AAN fast
# integer IDCT. No third-party code.

ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
    44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47,
    55, 62, 63,
)


class BitReader:
    """JPEG entropy bit stream: 0xFF00 unstuffs to 0xFF, restart
    markers (0xFFD0-0xFFD7) are skipped by the caller."""

    __slots__ = ("data", "pos", "acc", "nbits")

    def __init__(self, data, pos):
        self.data = data
        self.pos = pos
        self.acc = 0
        self.nbits = 0

    def _fill(self):
        d = self.data
        p = self.pos
        while self.nbits <= 24 and p < len(d):
            b = d[p]
            p += 1
            if b == 0xFF:
                if p < len(d) and d[p] == 0x00:
                    p += 1
                elif p < len(d) and 0xD0 <= d[p] <= 0xD7:
                    # restart marker: realign to the next byte
                    p += 1
                    self.nbits = 0
                    self.acc = 0
                    self.pos = p
                    return
                else:
                    # end of scan data; pad with ones
                    self.acc = (self.acc << 8) | 0xFF
                    self.nbits += 8
                    self.pos = p
                    return
            self.acc = (self.acc << 8) | b
            self.nbits += 8
        self.pos = p

    def bits(self, n):
        if n == 0:
            return 0
        if self.nbits < n:
            self._fill()
        if self.nbits < n:  # ran out: pad with 1 bits
            v = (self.acc << (n - self.nbits)) | ((1 << (n - self.nbits)) - 1)
            self.nbits = 0
            self.acc = 0
            return v & ((1 << n) - 1)
        self.nbits -= n
        v = (self.acc >> self.nbits) & ((1 << n) - 1)
        self.acc &= (1 << self.nbits) - 1
        return v

    def bit(self):
        return self.bits(1)


class HuffTable:
    """Canonical Huffman table with a fast lookup for short codes."""

    __slots__ = ("counts", "symbols", "lookup", "maxlen")

    def __init__(self, counts, symbols):
        self.counts = counts
        self.symbols = symbols
        self.maxlen = 16
        # 8-bit fast lookup: (length, symbol) or None
        self.lookup = [None] * 256
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                sym = symbols[k]
                k += 1
                if length <= 8:
                    base = code << (8 - length)
                    for j in range(1 << (8 - length)):
                        self.lookup[base | j] = (length, sym)
                code += 1
            code <<= 1

    def decode(self, br):
        # try the 8-bit fast path
        if br.nbits < 8:
            br._fill()
        if br.nbits >= 8:
            peek = (br.acc >> (br.nbits - 8)) & 0xFF
            hit = self.lookup[peek]
            if hit is not None:
                br.bits(hit[0])
                return hit[1]
        # slow path: walk the canonical code bit by bit
        code = 0
        first = 0
        index = 0
        for length in range(1, 17):
            code = (code << 1) | br.bit()
            cnt = self.counts[length - 1]
            if code - first < cnt:
                return self.symbols[index + (code - first)]
            index += cnt
            first = (first + cnt) << 1
        return 0


def _extend(v, n):
    """JPEG sign extension for a received value of n bits."""
    if n == 0:
        return 0
    if v < (1 << (n - 1)):
        return v - (1 << n) + 1
    return v


# 2-D IDCT by direct cosine matrix (separable, exact by
# construction). Precomputed once: M[k][n] folds in the 1/sqrt(2)
# DC factor and the 1/4 scale of the JPEG inverse DCT, so
# out[n] = sum_k M[k][n] * X[k].
_COS = []
for _k in range(8):
    _a = (1.0 / math.sqrt(2.0)) if _k == 0 else 1.0
    _COS.append([0.5 * _a * math.cos((2 * _n + 1) * _k * math.pi / 16.0)
                 for _n in range(8)])


def _idct_1d(row):
    """1-D inverse DCT of 8 dequantised coefficients."""
    if row[1] == 0 and row[2] == 0 and row[3] == 0 and row[4] == 0 \
            and row[5] == 0 and row[6] == 0 and row[7] == 0:
        v = row[0] * _COS[0][0]
        return [v] * 8
    c0, c1, c2, c3, c4, c5, c6, c7 = _COS
    x0, x1, x2, x3, x4, x5, x6, x7 = row
    return [
        x0 * c0[0] + x1 * c1[0] + x2 * c2[0] + x3 * c3[0]
        + x4 * c4[0] + x5 * c5[0] + x6 * c6[0] + x7 * c7[0],
        x0 * c0[1] + x1 * c1[1] + x2 * c2[1] + x3 * c3[1]
        + x4 * c4[1] + x5 * c5[1] + x6 * c6[1] + x7 * c7[1],
        x0 * c0[2] + x1 * c1[2] + x2 * c2[2] + x3 * c3[2]
        + x4 * c4[2] + x5 * c5[2] + x6 * c6[2] + x7 * c7[2],
        x0 * c0[3] + x1 * c1[3] + x2 * c2[3] + x3 * c3[3]
        + x4 * c4[3] + x5 * c5[3] + x6 * c6[3] + x7 * c7[3],
        x0 * c0[4] + x1 * c1[4] + x2 * c2[4] + x3 * c3[4]
        + x4 * c4[4] + x5 * c5[4] + x6 * c6[4] + x7 * c7[4],
        x0 * c0[5] + x1 * c1[5] + x2 * c2[5] + x3 * c3[5]
        + x4 * c4[5] + x5 * c5[5] + x6 * c6[5] + x7 * c7[5],
        x0 * c0[6] + x1 * c1[6] + x2 * c2[6] + x3 * c3[6]
        + x4 * c4[6] + x5 * c5[6] + x6 * c6[6] + x7 * c7[6],
        x0 * c0[7] + x1 * c1[7] + x2 * c2[7] + x3 * c3[7]
        + x4 * c4[7] + x5 * c5[7] + x6 * c6[7] + x7 * c7[7],
    ]


def idct_2d(block):
    """2-D inverse DCT on a flat 64-coefficient block (row-major,
    zig-zag already undone). Returns 64 spatial values."""
    rows = [_idct_1d(block[r * 8:(r + 1) * 8]) for r in range(8)]
    out = [0.0] * 64
    for c in range(8):
        col = _idct_1d([rows[r][c] for r in range(8)])
        for r in range(8):
            out[r * 8 + c] = col[r]
    return out


def _read_u16(data, pos):
    return (data[pos] << 8) | data[pos + 1]


class JpegImage:
    """One decoded baseline JPEG: full-resolution luma plane and
    block-resolution chroma planes (Cb/Cr DC values only)."""

    __slots__ = ("width", "height", "y", "cb", "cr", "cb_bw", "cb_bh")

    def __init__(self, width, height, y, cb, cr, cb_bw, cb_bh):
        self.width = width
        self.height = height
        self.y = y          # list of rows, each a list of ints 0..255
        self.cb = cb        # flat list, block resolution, signed
        self.cr = cr
        self.cb_bw = cb_bw
        self.cb_bh = cb_bh


def decode_jpeg(data):
    """Decode a baseline (SOF0) JPEG. Returns a JpegImage.
    Raises ValueError on anything unsupported (progressive, etc.)."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG")
    pos = 2
    quant = {}          # table id -> 64 dequant values (zig-zag order)
    dc_tables = {}
    ac_tables = {}
    width = height = 0
    comps = []          # (id, h_samp, v_samp, quant_id)
    scan = None
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = _read_u16(data, pos + 2)
        seg = data[pos + 4:pos + 2 + length]
        if marker == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                tq = seg[p]
                prec = tq >> 4
                tid = tq & 0x0F
                vals = []
                for k in range(64):
                    if prec == 0:
                        vals.append(seg[p + 1 + k])
                    else:
                        vals.append(_read_u16(seg, p + 1 + 2 * k))
                quant[tid] = vals
                p += 1 + (64 if prec == 0 else 128)
        elif marker == 0xC4:  # DHT
            p = 0
            while p < len(seg):
                tc = seg[p]
                cls = tc >> 4
                tid = tc & 0x0F
                counts = list(seg[p + 1:p + 17])
                total = sum(counts)
                symbols = list(seg[p + 17:p + 17 + total])
                table = HuffTable(counts, symbols)
                if cls == 0:
                    dc_tables[tid] = table
                else:
                    ac_tables[tid] = table
                p += 17 + total
        elif marker == 0xC0:  # SOF0 (baseline)
            height = _read_u16(seg, 1)
            width = _read_u16(seg, 3)
            ncomp = seg[5]
            comps = []
            for k in range(ncomp):
                cid = seg[6 + 3 * k]
                samp = seg[7 + 3 * k]
                tq = seg[8 + 3 * k]
                comps.append((cid, samp >> 4, samp & 0x0F, tq))
        elif marker == 0xDA:  # SOS: scan data follows
            ns = seg[0]
            scan = []
            for k in range(ns):
                cid = seg[1 + 2 * k]
                tt = seg[2 + 2 * k]
                scan.append((cid, tt >> 4, tt & 0x0F))
            pos = pos + 2 + length
            break
        pos += 2 + length
    if scan is None or not comps:
        raise ValueError("no scan found")
    comp_by_id = {c[0]: c for c in comps}
    # MCU geometry from the maximum sampling factors.
    hmax = max(c[1] for c in comps)
    vmax = max(c[2] for c in comps)
    mcu_w = 8 * hmax
    mcu_h = 8 * vmax
    mcus_x = (width + mcu_w - 1) // mcu_w
    mcus_y = (height + mcu_h - 1) // mcu_h

    br = BitReader(data, pos)
    # Per-component state.
    dc_pred = {cid: 0 for cid, _, _ in scan}
    # Output planes.
    y_plane = [[0] * width for _ in range(height)]
    chroma = {}   # cid -> (flat DC list, blocks_x, blocks_y)
    for cid, hs, vs, _tq in comps:
        if cid != 1:
            bw = (width * hs + 8 * hmax - 1) // (8 * hmax)
            bh = (height * vs + 8 * vmax - 1) // (8 * vmax)
            chroma[cid] = ([0] * (bw * bh), bw, bh)
    qt_of = {c[0]: quant[c[3]] for c in comps}

    for my in range(mcus_y):
        for mx in range(mcus_x):
            for cid, tdc, tac in scan:
                _cid, hs, vs, _tq = comp_by_id[cid]
                qtab = qt_of[cid]
                dct = dc_tables[tdc]
                act = ac_tables[tac]
                for by in range(vs):
                    for bx in range(hs):
                        # --- one 8x8 block: DC then AC ---
                        size = dct.decode(br)
                        diff = _extend(br.bits(size), size) if size else 0
                        dc_pred[cid] += diff
                        coef = [0] * 64
                        coef[0] = dc_pred[cid] * qtab[0]
                        k = 1
                        while k < 64:
                            sym = act.decode(br)
                            if sym == 0x00:  # EOB
                                break
                            run = sym >> 4
                            size = sym & 0x0F
                            if sym == 0xF0:  # ZRL
                                k += 16
                                continue
                            k += run
                            if k >= 64:
                                break
                            val = _extend(br.bits(size), size) if size else 0
                            coef[ZIGZAG[k]] = val * qtab[k]
                            k += 1
                        if cid == 1:
                            # full IDCT for luma
                            px = idct_2d(coef)
                            ox = (mx * hmax + bx) * 8
                            oy = (my * vmax + by) * 8
                            for r in range(8):
                                yy = oy + r
                                if yy >= height:
                                    break
                                row = y_plane[yy]
                                base = r * 8
                                for c in range(8):
                                    xx = ox + c
                                    if xx >= width:
                                        break
                                    v = int(px[base + c] + 128.5)
                                    row[xx] = 0 if v < 0 else (255 if v > 255 else v)
                        else:
                            # chroma: DC value only
                            arr, bw, bh = chroma[cid]
                            cx = mx * hs + bx
                            cy = my * vs + by
                            if cx < bw and cy < bh:
                                arr[cy * bw + cx] = int(dc_pred[cid] * qtab[0] / 8.0) + 128
    cb = cr = None
    cb_bw = cb_bh = 0
    for cid, (arr, bw, bh) in chroma.items():
        if cid == 2:
            cb, cb_bw, cb_bh = arr, bw, bh
        elif cid == 3:
            cr = arr
    if cb is None:
        cb = [128] * ((width + 7) // 8 * ((height + 7) // 8))
        cb_bw = (width + 7) // 8
        cb_bh = (height + 7) // 8
    if cr is None:
        cr = [128] * len(cb)
    return JpegImage(width, height, y_plane, cb, cr, cb_bw, cb_bh)


# ------------------------------------------------------------- features ----
def _clamp255(v):
    return 0 if v < 0 else (255 if v > 255 else v)


def extract_features(img):
    """Colour/texture signals from a decoded photo, on a grid about
    320 px wide. Orientation-aware: the sky side is the bluer half
    (several photos are flipped), the ground side the other one."""
    w, h = img.width, img.height
    step = max(1, w // 320)
    gw = (w + step - 1) // step
    gh = (h + step - 1) // step
    # Build an RGB grid from Y (full res) + nearest chroma block.
    grid = []
    cb, cr = img.cb, img.cr
    cbw, cbh = img.cb_bw, img.cb_bh
    yplane = img.y
    for gy in range(gh):
        y = min(h - 1, gy * step)
        cy = y * cbh // h
        yrow = yplane[y]
        row = []
        for gx in range(gw):
            x = min(w - 1, gx * step)
            cx = x * cbw // w
            cbi = cb[cy * cbw + cx] - 128
            cri = cr[cy * cbw + cx] - 128
            yy = yrow[x]
            r = _clamp255(int(yy + 1.402 * cri))
            g = _clamp255(int(yy - 0.34414 * cbi - 0.71414 * cri))
            b = _clamp255(int(yy + 1.772 * cbi))
            row.append((r, g, b))
        grid.append(row)

    n = gw * gh
    # Per-pixel predicate counts.
    yellow = green_l = green_s = arid = redsoil = skyblue = 0
    white = 0
    yrows = [0] * gh
    ytot = 0
    yxsum = 0
    rsum = gsum = bsum = 0
    edge = 0
    edgen = 0
    top_blue = bot_blue = 0
    top_n = bot_n = 0
    for gy in range(gh):
        row = grid[gy]
        for gx in range(gw):
            r, g, b = row[gx]
            rsum += r
            gsum += g
            bsum += b
            if r > 165 and g > 140 and b < 120 and r - b > 70 \
                    and g - b > 55 and r - g < 60:
                yellow += 1
                yrows[gy] += 1
                ytot += 1
                yxsum += gx
            if g >= r - 5 and g >= b - 5 and g > 55 and (g - r) + (g - b) > 6:
                green_l += 1
            if g > r + 8 and g > b + 8 and g > 60:
                green_s += 1
            if r > 130 and r >= g >= b and r - b > 40:
                arid += 1
            if r > 95 and r - g > 22 and r - b > 45:
                redsoil += 1
            if b > r + 25 and b > 110:
                skyblue += 1
                if gy < gh // 2:
                    top_blue += 1
                else:
                    bot_blue += 1
            if gy < gh // 2:
                top_n += 1
            else:
                bot_n += 1
            if r > 205 and g > 205 and b > 205 and max(r, g, b) - min(r, g, b) < 22:
                white += 1
            if gx + 1 < gw:
                r2, g2, b2 = row[gx + 1]
                edge += abs(r - r2) + abs(g - g2) + abs(b - b2)
                edgen += 1
    # Yellow-line concentration: a painted centre line is a thin
    # diagonal, so rows alone do not concentrate it; use the share
    # of yellow pixels in the densest 3-row band as a weak check.
    yline = 0.0
    if ytot:
        for gy in range(gh - 2):
            yline = max(yline, (yrows[gy] + yrows[gy + 1] + yrows[gy + 2]) / ytot)
    yx = (yxsum / ytot / gw) if ytot else 0.5
    # Ground half = the less-blue half (photos are sometimes flipped).
    sky_top = top_blue / max(1, top_n)
    sky_bot = bot_blue / max(1, bot_n)
    ground_is_top = sky_bot > sky_top
    # White fraction on the ground half (snow on the ground, not cloud).
    gw_count = 0
    gw_n = 0
    gg_count = 0
    gg_n = 0
    for gy in range(gh):
        is_ground = (gy < gh // 2) == ground_is_top
        for gx in range(gw):
            r, g, b = grid[gy][gx]
            if is_ground:
                gw_n += 1
                if r > 205 and g > 205 and b > 205 and max(r, g, b) - min(r, g, b) < 22:
                    gw_count += 1
                if g >= r - 5 and g >= b - 5 and g > 55 and (g - r) + (g - b) > 6:
                    gg_count += 1
                gg_n += 1
    return {
        "yellow": yellow / n,
        "yline": yline,
        "yx": yx,
        "greenL": green_l / n,
        "greenS": green_s / n,
        "arid": arid / n,
        "redsoil": redsoil / n,
        "skyblue": skyblue / n,
        "white": white / n,
        "snow_ground": gw_count / max(1, gw_n),
        "green_ground": gg_count / max(1, gg_n),
        "edge": edge / max(1, edgen),
        "warm": (rsum - bsum) / n,
        "bright": (rsum + gsum + bsum) / (3.0 * n),
    }


# ----------------------------------------------------------- classifier ----
# Region anchors: (name, lat, lon, prior). Priors roughly follow the
# practice pool's country mix and the OSV5M contributor mix: the
# far north and the Americas dominate, the long tail is thin.
ANCHORS = [
    ("RUw", 56, 38, 1.00), ("RUe", 55, 100, 0.15), ("CA", 54, -105, 0.80),
    ("US", 39, -98, 0.70), ("MX", 23, -102, 0.55), ("BR", -10, -52, 0.80),
    ("AR", -35, -65, 0.30), ("CL", -31, -71, 0.30), ("NZ", -42, 173, 0.30),
    ("JP", 36, 138, 0.30), ("KR", 36, 128, 0.20), ("CN", 31, 106, 0.25),
    ("IN", 21, 78, 0.30), ("ID", -3, 117, 0.25), ("PG", -6, 147, 0.20),
    ("AU", -25, 134, 0.25), ("NEU", 62, 20, 0.60), ("WEU", 47, 7, 0.65),
    ("CAS", 46, 70, 0.15), ("ME", 32, 48, 0.30), ("ARAB", 22, 52, 0.20),
    ("AFW", 7, 2, 0.40), ("AFE", -3, 37, 0.30), ("AFS", -19, 30, 0.30),
    ("PAC", -9, 160, 0.15),
]
_AMER = {"US", "MX", "BR", "AR", "CL", "CA"}
_LEFT = {"JP", "IN", "ID", "AU", "NZ", "PG", "PAC"}
_TROP = {"BR", "MX", "IN", "ID", "PG", "AFW", "AFE", "AFS", "PAC"}
_ARID = {"CL", "AR", "AU", "CAS", "ME", "ARAB", "US", "MX"}
_NORTH = {"RUw", "RUe", "CA", "NEU"}
_FOREST = {"RUw", "RUe", "CA", "NEU", "BR", "US", "CN"}
_AMAP = {name: (la, lo) for name, la, lo, _p in ANCHORS}

# Tuned on the kit's 75 practice photos: coarse, round weights,
# deliberately conservative so weak evidence cannot throw a photo
# across the world. Green foliage alone does not separate the
# tropics from boreal summer (both are green), so the green rules
# carry zero weight; the signals that survived tuning are aridity
# (southern/dry), big blue sky (Americas), warm light (eastern
# dry belt), yellow road paint/dirt (southern), and snow on the
# ground at a strict threshold (far north).
W_YEL = 1.0
W_ARID = 2.0
W_GREEN = 0.0
W_GREENL = 0.0
W_SNOW = 0.9
W_SKY = 1.2
W_EDGE = 0.3
W_WARM = 0.8
W_PRIOR = 1.0
TEMP = 1.5


def _scores(ft):
    s = {name: W_PRIOR * math.log(p) for name, _la, _lo, p in ANCHORS}

    def boost(names, v):
        for nm in names:
            if nm in s:
                s[nm] += v

    if ft["yellow"] > 0.002:
        # Yellow road paint (and sun-baked dirt, which is also a
        # southern signal): the Americas and New Zealand paint
        # centre lines yellow; Europe and most of Asia do not.
        boost(_AMER | {"NZ"}, W_YEL)
        boost({"WEU", "NEU", "JP", "KR", "CN", "IN", "ID", "ME", "CAS",
               "PAC", "ARAB"}, -W_YEL * 0.6)
        if ft["yx"] < 0.47:
            boost({"US", "MX", "BR", "AR", "CL", "CA"}, 0.4)
            boost(_LEFT, -0.4)
        elif ft["yx"] > 0.53:
            boost(_LEFT, 0.5)
            boost({"US", "MX", "BR", "AR", "CL", "CA"}, -0.3)
    if ft["arid"] > 0.06 or ft["redsoil"] > 0.05:
        boost(_ARID | {"AFW", "AFE", "AFS"}, W_ARID)
        boost(_TROP | {"NEU", "JP", "KR", "CN", "RUw", "RUe", "CA", "NZ"},
              -W_ARID * 0.6)
    if ft["greenS"] > 0.10:
        boost(_TROP, W_GREEN)
        boost({"CAS", "ME", "ARAB", "CL", "AR"}, -W_GREEN * 0.7)
    if ft["greenL"] > 0.35:
        boost(_FOREST | _TROP, W_GREENL)
        boost({"CAS", "ME", "ARAB", "AU"}, -W_GREENL * 0.8)
    if ft["snow_ground"] > 0.15:
        # Strict: only real snow on the ground clears this; cloud
        # and bright road stay below it.
        boost(_NORTH, W_SNOW)
        boost(_TROP | {"ARAB", "ME", "AU", "BR", "MX"}, -W_SNOW * 0.5)
    if ft["warm"] > 0:
        # Warm (red-shifted) light: the eastern dry belt.
        boost({"ME", "CAS", "ARAB", "AFW", "AFE"}, W_WARM)
        boost({"CA", "US", "NEU", "WEU"}, -W_WARM * 0.5)
    if ft["skyblue"] > 0.35 and ft["greenL"] < 0.25 and ft["arid"] < 0.05:
        boost({"US", "RUw", "RUe", "CA", "CAS", "WEU", "NEU"}, W_SKY)
    if ft["edge"] > 28:
        boost({"WEU", "JP", "IN", "ME", "KR"}, W_EDGE)
    return s


def place(ft, seed):
    """Posterior-weighted geographic mean over the anchors.
    The seed is accepted for API stability; the placement is a
    deterministic function of the pixels, so the seed is unused."""
    s = _scores(ft)
    top = max(s.values())
    vec_x = vec_y = vec_z = 0.0
    tot = 0.0
    for name, (la, lo) in _AMAP.items():
        wgt = math.exp((s[name] - top) / TEMP)
        tot += wgt
        la_r = math.radians(la)
        lo_r = math.radians(lo)
        vec_x += wgt * math.cos(la_r) * math.cos(lo_r)
        vec_y += wgt * math.cos(la_r) * math.sin(lo_r)
        vec_z += wgt * math.sin(la_r)
    vec_x /= tot
    vec_y /= tot
    vec_z /= tot
    lat = math.degrees(math.atan2(vec_z, math.hypot(vec_x, vec_y)))
    lon = math.degrees(math.atan2(vec_y, vec_x))
    return round(lat, 6), round(lon, 6)


# ----------------------------------------------------------------- env -----
def _http(method, url, body=None):
    req = urllib.request.Request(url, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def predict_file(path):
    """Decodes one JPEG file and returns its lat/lon placement."""
    with open(path, "rb") as f:
        jpeg_bytes = f.read()
    decoded = decode_jpeg(jpeg_bytes)
    features = extract_features(decoded)
    return place(features, 0)

def run():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    seed_text = os.environ.get("MAKEFUN_TRIAL_SEED", "0")
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)  # noqa: F841 (seeded; kept for determinism)
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as fh:
                fh.read()
        except OSError:
            pass
    try:
        status, raw = _http("GET", env_url.rstrip("/") + "/photos")
    except Exception:
        return 1
    if status != 200:
        return 1
    listing = json.loads(raw.decode("utf-8"))
    photos = listing.get("photos", [])
    answers = []
    for photo in photos:
        pid = photo["id"]
        try:
            _st, data = _http("GET", env_url.rstrip("/") + photo["url"])
            img = decode_jpeg(data)
            ft = extract_features(img)
            lat, lon = place(ft, seed)
        except Exception:
            # Any failure still answers: the un-placed fallback is
            # the prior-heavy point, far better than unanswered.
            lat, lon = 56.0, 30.0
        lat = max(-90.0, min(90.0, lat))
        lon = max(-180.0, min(180.0, lon))
        answers.append({"id": pid, "lat": lat, "lon": lon})
    if answers:
        status, _raw = _http("POST", env_url.rstrip("/") + "/answers",
                             {"answers": answers})
        if status != 200:
            # Retry once, split in two halves (a 409 on one half
            # would only follow a partial earlier post, which we
            # never make; the split only helps an oversized body).
            half = len(answers) // 2
            _http("POST", env_url.rstrip("/") + "/answers",
                  {"answers": answers[:half]})
            _http("POST", env_url.rstrip("/") + "/answers",
                  {"answers": answers[half:]})
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(VERSION)
        sys.exit(0)
    elif len(sys.argv) == 3 and sys.argv[1] == "--photo":
        lat, lon = predict_file(sys.argv[2])
        print(f"{lat},{lon}")
    else:
        sys.exit(run())

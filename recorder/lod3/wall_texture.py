"""壁・窓・扉に貼る写真の質感（テクスチャ）を、公開版の写真から面ごとに作る（11/19）。

先行の方法（`D:\\PLATEAU\\lod3\\research\\texture_prior_art.md`）から取り入れたこと：
- 面ごとに合成する。写真は足切りしてから使う（使える所の面積、位置合わせの誤差、外壁の読み係が「ずれ」とした写真）。
  国交省の技術資料 No.139 は 1 面に 1 枚の写真を射影変換するだけで、複数の写真の合成・遮蔽の除去・色合わせは課題として残していた。
- 3 枚以上の写真が重なる画素は、画素ごとの中央値で合成する（木・人・他の棟の遮蔽を消す。Ortin & Remondino）。
- 1〜2 枚しか無い画素は、点数の高い写真を 1 枚選ぶ（写真ごとに大きな領域で貼り、継ぎ目を減らす。Waechter ほか 2014 の考え）。
- 写真ごとの明るさ・色の差は、重なる所で基準と比べた 3 色の倍率で合わせる（Waechter ほかの全体の色合わせを簡単にした物）。
- どの写真にも写っていない所は無理に埋めず、その面の写った所の色の中央値で塗る（未観測として記録する）。
- 出力は PLATEAU 標準製品仕様書のテクスチャの決まり：JPEG、1 辺は 2 の累乗で 2048 px まで、余白は灰色 (90,90,90)、
  app:ParameterizedTexture・theme rgbTexture、見た目の出典のコード 4（手持ちカメラで撮った写真）。
- 面ごとに、使った写真の枚数・実効の解像度（px/m）・写った割合・位置合わせの誤差を数値で残す。

使う写真は公開版（掲示物・人名・顔・ナンバーを灰色の箱で塗った版）だけ。使える所のマスクは寸法用の面の画像
（out/facade/<棟>/<面>/<写真>_mask.png、形だけで写真の中身は無い）から取る。
合成は見た目だけに使う。寸法の根拠にはしない（CLAUDE.md の決まり 3）。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

PPM = 40.0                  # 面の画像の解像度（recorder facade views と同じ）
MAX_VIEWS = 12              # 1 面に使う写真の上限（点数の高い順）
MAX_RMS_PX = 8.0            # 位置合わせの誤差（再投影の RMS、元の写真の px）がこれを超える写真は使わない
MIN_USABLE_M2 = 4.0
MEDIAN_MIN = 3              # この枚数以上が重なる画素は中央値
ERODE_PX = 2                # マスクの縁を削る（縁の引き伸ばされた画素を避ける）
GAIN_CLIP = (0.6, 1.6)
MAX_SIDE = 2048             # 標準製品仕様書：1 辺 2048 px が基本（4096 まで）
PAD_RGB = (90, 90, 90)      # 標準製品仕様書：アトラスの背景は黒か灰色 (90,90,90)
FACADE = Path("out") / "facade"
PUBLIC = Path("out") / "facade_public"
RESULT_FILES = ("detections.json", "fused.json", "refined.json", "final.json", "compare.json")


PUBLIC_LONG = 1600          # 公開版の写真の長辺（px）
_IDX: dict = {}


def _public_ratio(photo: str, root: Path) -> float:
    """公開版の写真の縮小の比（公開版の長辺 ÷ 元の写真の長辺）。分からなければ 1。"""
    key = str(Path(root).resolve())
    if key not in _IDX:
        try:
            d = json.loads((Path(root) / "data" / "photos" / "index.json").read_text(encoding="utf-8"))
            d = d["photos"] if isinstance(d, dict) else d
            _IDX[key] = {p["id"]: p for p in d if isinstance(p, dict)}
        except (OSError, ValueError):
            _IDX[key] = {}
    p = _IDX[key].get(photo) or {}
    long = max(int(p.get("width") or 0), int(p.get("height") or 0))
    return min(1.0, PUBLIC_LONG / long) if long else 1.0


def _rms(meta: dict) -> float | None:
    reg = meta.get("registration") or {}
    for k in ("reproj_rms_px", "pose_holdout_px", "reproj_median_px"):
        v = reg.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def face_views(bid: str, fid: str, gaiheki: dict | None = None, root: Path = Path(".")) -> tuple[list[dict], list[dict]]:
    """面の公開版の画像を集め、足切りする。戻り値 (使う物, 外した物と理由)。"""
    import cv2

    pub_dir, fac_dir = root / PUBLIC / bid / fid, root / FACADE / bid / fid
    bad = {}
    for a in (gaiheki or {}).get("photo_alignment") or []:
        if not a.get("usable", True) or abs(a.get("shift_m") or 0) > 0.5:
            bad[a["photo"]] = f"外壁の読み係が位置合わせのずれを見つけた（{a.get('shift_m')} m）"
    use, skipped = [], []
    for mj in sorted(pub_dir.glob("*.json")) if pub_dir.exists() else []:
        if mj.name in RESULT_FILES:
            continue
        meta = json.loads(mj.read_text(encoding="utf-8"))
        ph = meta.get("photo")
        why = None
        rms = _rms(meta)
        if ph in bad:
            why = bad[ph]
        elif (meta.get("usable_m2") or 0) < MIN_USABLE_M2:
            why = f"使える所が {meta.get('usable_m2')} m² と小さい"
        elif rms is not None and rms > MAX_RMS_PX:
            why = f"位置合わせの誤差 {rms:.1f} px が {MAX_RMS_PX:g} px を超える"
        img_p, mask_p = pub_dir / meta.get("image", ""), fac_dir / meta.get("mask", "")
        if why is None and not (img_p.is_file() and mask_p.is_file()):
            why = "画像かマスクが無い"
        if why:
            skipped.append({"photo": ph, "why": why})
            continue
        img = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(mask_p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
            skipped.append({"photo": ph, "why": "画像とマスクの大きさが合わない"})
            continue
        valid = mask > 0
        red_p = pub_dir / (meta.get("redact_mask") or "")
        n_red = 0
        if meta.get("redact_mask") and red_p.is_file():          # 個人情報を塗った箱は「写っていない」扱い
            red = cv2.imdecode(np.fromfile(str(red_p), np.uint8), cv2.IMREAD_GRAYSCALE)
            if red is not None and red.shape == valid.shape:
                n_red = int((valid & (red > 0)).sum())
                valid &= ~(red > 0)
        # 公開版の写真は縮小してある（長辺 1600 px）ので、面の上の実効の解像度は元の写真の値 × 縮小の比
        dens = float(meta.get("median_photo_px_per_m") or 0) * _public_ratio(ph, root)
        score = dens / (1.0 + (rms or 0.0) / 4.0)      # 解像度が高く、位置合わせの誤差が小さい写真ほど上
        use.append({"photo": ph, "meta": meta, "img": img, "mask": valid, "rms": rms, "dens": dens, "score": score,
                    "redacted_m2": round(n_red / (PPM * PPM), 2)})
    use.sort(key=lambda v: -v["score"])
    for v in use[MAX_VIEWS:]:
        skipped.append({"photo": v["photo"], "why": f"点数が上位 {MAX_VIEWS} 枚に入らない"})
    return use[:MAX_VIEWS], skipped


def fill_unseen(out: np.ndarray, seen: np.ndarray, pitch_px: float | None) -> tuple[np.ndarray, dict]:
    """どの写真にも写っていない画素を埋める（見た目だけ。報告に「推定で埋めた割合」を残す）。
    1. 柱間のくり返しがあれば、同じ行の ±1〜3 柱間となりの写った画素を写す（立面のくり返しで遮蔽を補う方法。
       垂れ幕・木で隠れた所に、となりの柱間の窓と壁を置く）。
    2. 残りは周りからなめらかに埋める（cv2.inpaint、Telea）。"""
    import cv2

    H, W = seen.shape
    filled = seen.copy()
    n_rep = 0
    if pitch_px and pitch_px > 8:
        for k in (1, -1, 2, -2, 3, -3):
            dx = int(round(k * pitch_px))
            if abs(dx) >= W:
                continue
            need = ~filled
            if not need.any():
                break
            src = np.zeros_like(seen)
            if dx > 0:
                src[:, :-dx] = seen[:, dx:]
            else:
                src[:, -dx:] = seen[:, :dx]
            take = need & src
            ys, xs = np.nonzero(take)
            out[ys, xs] = out[ys, np.clip(xs + dx, 0, W - 1)]
            filled |= take
            n_rep += int(take.sum())
    rest = ~filled
    n_inp = int(rest.sum())
    if n_inp:
        out = cv2.inpaint(out, rest.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    total = H * W
    return out, {"filled_by_repeat": round(n_rep / total, 3), "filled_by_inpaint": round(n_inp / total, 3)}


def compose(rect: tuple[float, float, float, float], views: list[dict], ppm: float = PPM,
            pitch_m: float | None = None) -> tuple[np.ndarray | None, dict]:
    """面の範囲（s0, s1, z0, z1、m）の画像を合成する。戻り値 (BGR 画像か None, 記録)。
    pitch_m（柱間）があれば、写っていない所をとなりの柱間から写して埋める。"""
    import cv2

    s0, s1, z0, z1 = rect
    W, H = max(1, int(round((s1 - s0) * ppm))), max(1, int(round((z1 - z0) * ppm)))
    info = {"size_px": [W, H], "px_per_m": ppm, "views": len(views)}
    if not views:
        return None, {**info, "why": "使える写真が無い"}
    k = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), np.uint8)
    layers, valids = [], []
    for v in views:
        ox = int(round((float(v["meta"]["origin_m"]["s"]) - s0) * ppm))
        oy = int(round((z1 - float(v["meta"]["origin_m"]["z_top"])) * ppm))
        img, m = v["img"], cv2.erode(v["mask"].astype(np.uint8), k) > 0
        L = np.zeros((H, W, 3), np.uint8)
        M = np.zeros((H, W), bool)
        h, w = img.shape[:2]
        xa, ya, xb, yb = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
        if xb > xa and yb > ya:
            L[ya:yb, xa:xb] = img[ya - oy:yb - oy, xa - ox:xb - ox]
            M[ya:yb, xa:xb] = m[ya - oy:yb - oy, xa - ox:xb - ox]
        layers.append(L)
        valids.append(M)
    S = np.stack(layers).astype(np.float32)            # (n, H, W, 3)
    V = np.stack(valids)                               # (n, H, W)
    n_valid = V.sum(0)
    if not n_valid.any():
        return None, {**info, "why": "面の範囲に写っていない"}

    def merge(S):
        out = np.zeros((H, W, 3), np.float32)
        best = np.full((H, W), -1)
        for i in range(len(views)):                    # 点数の高い順に、まだ埋まっていない所へ
            take = V[i] & (best < 0)
            best[take] = i
        has = best >= 0
        idx = np.where(has, best, 0)
        out[has] = S[idx[has], np.nonzero(has)[0], np.nonzero(has)[1]]
        many = n_valid >= MEDIAN_MIN
        if many.any():
            for r0 in range(0, H, 128):                 # 行の帯ごとに中央値（メモリを抑える）
                r1 = min(H, r0 + 128)
                sel = many[r0:r1]
                if sel.any():
                    Sm = np.where(V[:, r0:r1, :, None], S[:, r0:r1], np.nan)
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)       # 写っていない画素の全 NaN（使わない）
                        med = np.nanmedian(Sm, axis=0)
                    out[r0:r1][sel] = med[sel]
        return out

    ref = merge(S)
    gains = []
    for i in range(len(views)):                         # 写真ごとの 3 色の倍率（重なる所で基準に合わせる）
        m = V[i] & (n_valid >= 2)
        if m.sum() < 200:
            gains.append([1.0, 1.0, 1.0])
            continue
        g = np.median(ref[m], axis=0) / np.maximum(np.median(S[i][m], axis=0), 1.0)
        gains.append(list(np.clip(g, *GAIN_CLIP)))
    S = np.clip(S * np.array(gains, np.float32)[:, None, None, :], 0, 255)
    out = merge(S)
    seen = n_valid > 0
    fill = np.median(out[seen], axis=0)
    out[~seen] = fill
    out8, fill_info = fill_unseen(out.astype(np.uint8), seen, pitch_m * ppm if pitch_m else None)
    out = out8.astype(np.float32)
    info.update(fill_info)
    info.update({
        "coverage": round(float(seen.mean()), 3),               # どれかの写真に写った割合
        "coverage_median": round(float((n_valid >= MEDIAN_MIN).mean()), 3),   # 3 枚以上が重なり、中央値で合成した割合
        "fill_bgr": [int(x) for x in fill],
        "used": [{"photo": v["photo"], "rms_px": v["rms"], "public_px_per_m": round(v["dens"], 1),
                  "redacted_m2": v.get("redacted_m2", 0), "gain_bgr": [round(float(x), 3) for x in g]} for v, g in zip(views, gains)],
        "effective_px_per_m": round(float(np.median([v["dens"] for v in views])), 1),
    })
    return out.astype(np.uint8), info


def to_pow2(img: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float, tuple[int, int]]:
    """1 辺を 2 の累乗にする（大きければ縮めてから、右と下を灰色で埋める）。戻り値 (画像, 縮尺, (幅, 高さ) の 2 の累乗)。"""
    import cv2

    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1.0:
        img = cv2.resize(img, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    P = lambda n: 1 << (int(n) - 1).bit_length()
    Wp, Hp = P(w), P(h)
    out = np.empty((Hp, Wp, 3), np.uint8)
    out[:] = PAD_RGB[::-1]
    out[:h, :w] = img
    return out, k, (Wp, Hp)


def tex_coords(P: np.ndarray, p0: np.ndarray, u: np.ndarray, rect, ppm: float, k: float, size: tuple[int, int]) -> list[tuple[float, float]]:
    """3 次元の点（棟の局所座標）→ 面の画像のテクスチャ座標 (u, v)。v は下から（CityGML の決まり）。"""
    s0, s1, z0, z1 = rect
    Wp, Hp = size
    out = []
    for p in P:
        s = float((np.asarray(p[:2]) - p0[:2]) @ u[:2])
        x = (s - s0) * ppm * k
        y = (z1 - float(p[2])) * ppm * k
        out.append((x / Wp, 1.0 - y / Hp))
    return out

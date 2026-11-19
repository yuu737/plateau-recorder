"""壁・窓・扉に貼る写真の質感（テクスチャ）を、公開版の写真から面ごとに作る（11/19）。

先行の方法（`D:\\PLATEAU\\lod3\\research\\texture_prior_art.md`）から取り入れたこと：
- 面ごとに合成する。写真は足切りしてから使う（使える所の面積、位置合わせの誤差、外壁の読み係が「ずれ」とした写真）。
  国交省の技術資料 No.139 は 1 面に 1 枚の写真を射影変換するだけで、複数の写真の合成・遮蔽の除去・色合わせは課題として残していた。
- 3 枚以上の写真が重なる画素は、画素ごとの中央値で合成する（木・人・他の棟の遮蔽を消す。Ortin & Remondino）。
- 1〜2 枚しか無い画素は、点数の高い写真を 1 枚選ぶ（写真ごとに大きな領域で貼り、継ぎ目を減らす。Waechter ほか 2014 の考え）。
- 写真ごとの明るさ・色の差は、重なる所で基準と比べた 3 色の倍率で合わせる（Waechter ほかの全体の色合わせを簡単にした物）。
- どの写真にも写っていない所は無理に埋めず、その面の写った所の色の中央値で塗る（未観測として記録する）。
- 出力は PLATEAU 標準製品仕様書のテクスチャの決まり：JPEG、1 辺は 2 の累乗で 2048 px が基本・大規模建築物は 4096 px まで、
  余白は灰色 (90,90,90)、app:ParameterizedTexture・theme rgbTexture、見た目の出典のコード 4（手持ちカメラで撮った写真）。
  面の画像は 80 px/m（1.25 cm/px）で作り、1 つの CityGML の分を 4096 px の画像（アトラス）に詰める（recorder/lod3/atlas.py）。
  1 辺が 4096 px に収まらない面（タワー75 の高さ 72 m の面など）だけ、収まるまで解像度を下げる。
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

PPM = 80.0                  # 質感の画像の解像度（1.25 cm/px）。公開版の面の画像（recorder facade public）も既定で 80 px/m
MAX_VIEWS = 12              # 1 面に使う写真の上限（点数の高い順）
MAX_RMS_PX = 8.0            # 位置合わせの誤差（再投影の RMS、元の写真の px）がこれを超える写真は使わない
MIN_USABLE_M2 = 4.0
MEDIAN_MIN = 3              # この枚数以上が重なる画素は中央値
ERODE_PX = 2                # マスクの縁を削る（縁の引き伸ばされた画素を避ける）
GAIN_CLIP = (0.6, 1.6)
MAX_SIDE = 4096             # 標準製品仕様書：1 辺 2048 px が基本、大規模建築物は 4096 px まで
PAD_RGB = (90, 90, 90)      # 標準製品仕様書：アトラスの背景は黒か灰色 (90,90,90)
FACADE = Path("out") / "facade"
PUBLIC = Path("out") / "facade_public"
RESULT_FILES = ("detections.json", "fused.json", "refined.json", "final.json", "compare.json")


PUBLIC_LONG = 1600          # 公開版の写真の長辺（px）
_IDX: dict = {}


def _public_ratio(photo: str, root: Path, public_long: int | None = None) -> float:
    """公開版の写真の縮小の比（公開版の長辺 ÷ 元の写真の長辺）。public_long は面の画像を作った時の公開版の長辺
    （元の解像度の公開版なら比は 1）。分からなければ 1600 px として計算する。"""
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
    return min(1.0, (public_long or PUBLIC_LONG) / long) if long else 1.0


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
        img_p = pub_dir / meta.get("image", "")
        mask_p = pub_dir / meta.get("mask", "")                 # 公開版と同じ解像度のマスク（recorder facade public）
        if not mask_p.is_file():
            mask_p = fac_dir / meta.get("mask", "")             # 無ければ窓の検出用（40 px/m）を拡大して使う
        if why is None and not (img_p.is_file() and mask_p.is_file()):
            why = "画像かマスクが無い"
        if why:
            skipped.append({"photo": ph, "why": why})
            continue
        img = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(mask_p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None and img.shape[:2] != mask.shape[:2]:
            r = img.shape[1] / mask.shape[1]
            if abs(img.shape[0] / mask.shape[0] - r) < 0.02:     # 解像度だけ違う（同じ範囲）なら最近傍で拡大
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
            skipped.append({"photo": ph, "why": "画像とマスクの大きさが合わない"})
            continue
        vppm = float(meta.get("px_per_m") or 40.0)
        valid = mask > 0
        red_p = pub_dir / (meta.get("redact_mask") or "")
        n_red = 0
        if meta.get("redact_mask") and red_p.is_file():          # 個人情報を塗った箱は「写っていない」扱い
            red = cv2.imdecode(np.fromfile(str(red_p), np.uint8), cv2.IMREAD_GRAYSCALE)
            if red is not None and red.shape == valid.shape:
                n_red = int((valid & (red > 0)).sum())
                valid &= ~(red > 0)
        # 公開版の写真は縮小してある（長辺 1600 px）ので、面の上の実効の解像度は元の写真の値 × 縮小の比
        dens = float(meta.get("median_photo_px_per_m") or 0) * _public_ratio(ph, root, meta.get("public_long"))
        score = dens / (1.0 + (rms or 0.0) / 4.0)      # 解像度が高く、位置合わせの誤差が小さい写真ほど上
        use.append({"photo": ph, "meta": meta, "img": img, "mask": valid, "rms": rms, "dens": dens, "score": score,
                    "ppm": vppm, "redacted_m2": round(n_red / (vppm * vppm), 2)})
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
    """面の範囲（s0, s1, z0, z1、m）の画像を ppm px/m で合成する。戻り値 (BGR 画像か None, 記録)。
    写真ごとの面の画像の解像度（meta の px_per_m）が ppm と違えば合わせてから重ねる。
    pitch_m（柱間）があれば、写っていない所をとなりの柱間から写して埋める。"""
    import cv2

    s0, s1, z0, z1 = rect
    W, H = max(1, int(round((s1 - s0) * ppm))), max(1, int(round((z1 - z0) * ppm)))
    info = {"size_px": [W, H], "px_per_m": round(ppm, 2), "views": len(views)}
    if not views:
        return None, {**info, "why": "使える写真が無い"}
    n = len(views)
    S = np.zeros((n, H, W, 3), np.uint8)                # 写真ごとの層（uint8 で持ち、帯ごとに小数にする）
    V = np.zeros((n, H, W), bool)
    for i, v in enumerate(views):
        img, m = v["img"], v["mask"].astype(np.uint8)
        r = ppm / float(v.get("ppm") or ppm)
        if abs(r - 1.0) > 1e-6:
            size = (max(1, int(round(img.shape[1] * r))), max(1, int(round(img.shape[0] * r))))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
            m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        e = max(1, int(round(ERODE_PX * ppm / 40.0)))   # マスクの縁を削る幅は 40 px/m で 2 px（＝5 cm）
        m = cv2.erode(m, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) > 0
        ox = int(round((float(v["meta"]["origin_m"]["s"]) - s0) * ppm))
        oy = int(round((z1 - float(v["meta"]["origin_m"]["z_top"])) * ppm))
        h, w = img.shape[:2]
        xa, ya, xb, yb = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
        if xb > xa and yb > ya:
            S[i, ya:yb, xa:xb] = img[ya - oy:yb - oy, xa - ox:xb - ox]
            V[i, ya:yb, xa:xb] = m[ya - oy:yb - oy, xa - ox:xb - ox]
    n_valid = V.sum(0)
    if not n_valid.any():
        return None, {**info, "why": "面の範囲に写っていない"}
    best = np.full((H, W), -1)
    for i in range(n):                                  # 点数の高い順に、まだ埋まっていない所へ
        best[V[i] & (best < 0)] = i
    many = n_valid >= MEDIAN_MIN

    def merge(gains: np.ndarray) -> np.ndarray:
        out = np.zeros((H, W, 3), np.float32)
        for r0 in range(0, H, 128):                     # 行の帯ごと（メモリを抑える）
            r1 = min(H, r0 + 128)
            Sb = np.clip(S[:, r0:r1].astype(np.float32) * gains[:, None, None, :], 0, 255)
            bb = best[r0:r1]
            ys, xs = np.nonzero(bb >= 0)
            ob = out[r0:r1]
            ob[ys, xs] = Sb[bb[ys, xs], ys, xs]
            sel = many[r0:r1]
            if sel.any():
                Sm = np.where(V[:, r0:r1, :, None], Sb, np.nan)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)       # 写っていない画素の全 NaN（使わない）
                    med = np.nanmedian(Sm, axis=0)
                ob[sel] = med[sel]
        return out

    ref = merge(np.ones((n, 3), np.float32))
    gains = []
    for i in range(n):                                  # 写真ごとの 3 色の倍率（重なる所で基準に合わせる）
        m = V[i] & (n_valid >= 2)
        if m.sum() < 200:
            gains.append([1.0, 1.0, 1.0])
            continue
        g = np.median(ref[m], axis=0) / np.maximum(np.median(S[i][m], axis=0), 1.0)
        gains.append(list(np.clip(g, *GAIN_CLIP)))
    out = merge(np.array(gains, np.float32))
    del ref
    seen = n_valid > 0
    fill = np.median(out[seen], axis=0)
    out[~seen] = fill
    out8, fill_info = fill_unseen(out.astype(np.uint8), seen, pitch_m * ppm if pitch_m else None)
    info.update(fill_info)
    info.update({
        "coverage": round(float(seen.mean()), 3),               # どれかの写真に写った割合
        "coverage_median": round(float(many.mean()), 3),        # 3 枚以上が重なり、中央値で合成した割合
        "fill_bgr": [int(x) for x in fill],
        "used": [{"photo": v["photo"], "rms_px": v["rms"], "public_px_per_m": round(v["dens"], 1),
                  "redacted_m2": v.get("redacted_m2", 0), "gain_bgr": [round(float(x), 3) for x in g]} for v, g in zip(views, gains)],
        "effective_px_per_m": round(float(np.median([v["dens"] for v in views])), 1),
    })
    return out8, info


def face_ppm(rect, ppm: float = PPM, max_side: int = MAX_SIDE, margin: int = 2) -> float:
    """面を ppm px/m で作った時に 1 辺（＋アトラスの余白）が max_side を超えるなら、収まる解像度に下げる。"""
    s0, s1, z0, z1 = rect
    longest = max(s1 - s0, z1 - z0, 1e-6)
    return min(ppm, (max_side - 2 * margin) / longest * 0.999)

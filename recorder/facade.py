"""写真 1 枚ずつを面の座標に正面化し、窓を拾い、写真どうしで突き合わせて値と σ を出す（PLATEAU 版の中心）。

なぜ合成しないか：campus-recorder は登録済みの写真を面に貼り重ねた合成（coverage_<面>_photo.png）を寸法の板に
していた。合成は写真ごとの位置合わせの誤差と手前の物（別の棟・木・掲示物）を混ぜ、どの写真から来た値かを消す。
ここでは写真ごとに別の画像のまま扱い、同じ窓を何枚が見たか・写真どうしでどれだけ食い違うかを、そのまま値の確かさにする。

流れ（`recorder facade …`）
  views    写真 1 枚 × 面 1 つ ごとに、面の座標（px_per_m、左上＝面の始点・上端）へ正面化した画像を作る。
           使えない所（面の外・斜めで引き伸ばされた所・手で書いた遮蔽・自分の棟の他の面・PLATEAU の他の棟の陰）は灰色に塗る。
           out/facade/<棟>/<面>/<写真>.jpg と .json（切り出しの原点、使えた面積、遮った棟、視点）。
  detect   各画像で窓の箱を拾う（Grounding DINO tiny、零ショット、Apache-2.0）。箱は面の m に直す。
  fuse     写真をまたいで同じ窓を束ね、縁ごとに中央値と、写真どうしのばらつき（標準偏差）から σ を出す。
           1 枚の写真にしか写らない窓は値を出すが σ は出さない（照合なし）。
  compare  棟 JSON の窓（設計係が柱割りから並べた物）と突き合わせ、ずれ・模型に無い窓・写真に無い窓を出す。

σ の決め方（1σ、m）：同じ縁を n 枚（n ≥ 2）が測った値の標準偏差 sd から σ = √(sd²/n + FLOOR²)。FLOOR は 1 画素
と検出の揺れの下限（0.05 m）。写真ごとの位置合わせは独立に解いているので、写真どうしのばらつきに位置合わせの誤差も入る。
同じ向きに偏る誤差（全部の写真に共通の縮尺の誤差など）は入らないので、答え合わせ（巻尺・距離計）で確かめる。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import coverage as CV
from . import drawing
from .registry import entries_for, expand_entries, face_top, load_registry, plane_to_world
from .schema import load_building

PX_PER_M = 40.0
PUBLIC_PX_PER_M = 80.0      # 公開版の面の画像（質感の元）の解像度。窓の検出用の 40 px/m の切り出しとマスクを倍にして使う
OUT_ROOT = Path("out") / "facade"
RESULT_FILES = ("detections.json", "fused.json", "refined.json", "final.json", "compare.json")   # 面のフォルダの、写真ごとでない結果
MIN_USABLE_M2 = 4.0          # これ未満しか使えない写真は面の画像を作らない
GRAY = (128, 128, 128)
FLOOR_M = 0.05               # σ の下限（1 画素 0.025 m と検出の揺れ）
MATCH_IOU = 0.3              # 写真をまたいで同じ窓とみなす重なり（面の m で）
BOX_LIMITS = {"w": (0.3, 12.0), "h": (0.3, 6.0)}     # 窓の箱として受け取る大きさ（m）
TILE_PX = 1000               # 検出の切り出しの一辺（px）。縦に長い面（塔）は縦に切る
TILE_OVERLAP = 0.25


# ------------------------------------------------------------------ 面の画像（写真 1 枚ずつ）

def face_rect(b: dict, f: dict) -> tuple[float, float, float, float] | None:
    L = CV._known_len(f)
    if L is None or not (f.get("from") and f.get("to")):
        return None
    g, _ = drawing.face_ground_line(b, f)
    return (0.0, float(L), float(g), float(face_top(b, f)))


def building_json(bid: str) -> Path:
    return Path("data") / "buildings" / f"{bid}.json"


def views(bid: str, faces: list[str] | None = None, ppm: float = PX_PER_M, out_root: Path | None = None,
          use_occlusion: bool = True, registry: dict | None = None, photos: list[str] | None = None) -> dict:
    """写真 1 枚 × 面 1 つ ごとに、面の座標へ正面化した画像を書く。photos を渡せばその写真だけ。
    窓の検出・照合は 40 px/m の画像を使うので、解像度を上げた画像（タイル・柱型の縁を読む用）は別のフォルダに書く
    （run_cli の --px-per-m。既定 out/facade_ppm<N>）。"""
    import cv2

    from .occlusion import Walls

    b = load_building(building_json(bid))
    reg = registry if registry is not None else load_registry()
    ents = expand_entries(entries_for(reg, bid), b)
    walls = Walls(b) if use_occlusion else None
    root = (out_root or OUT_ROOT) / bid
    res = {"building": bid, "made_at": datetime.now().isoformat(timespec="seconds"), "px_per_m": ppm,
           "occlusion": {"source": "PLATEAU LOD1（名古屋 2022 52365768）", "skipped_self": walls.skipped if walls else None},
           "faces": []}
    for f in b.get("faces", []):
        fid = f["id"]
        if faces and fid not in faces:
            continue
        rect = face_rect(b, f)
        fents = [e for e in ents if e["face"] == fid and (not photos or e["photo"] in photos)] if rect else []
        if not fents:
            continue
        S, Z = CV.pixel_grid(rect, ppm)
        fr = {"from": f["from"], "to": f["to"]}
        P = plane_to_world(fr, S, Z)
        A2, B2 = np.array(f["from"], float), np.array(f["to"], float)
        out = root / fid
        out.mkdir(parents=True, exist_ok=True)
        row = {"face": fid, "name": f.get("name"), "rect": [round(v, 3) for v in rect], "views": [], "skipped": []}
        for e in fents:
            tm = CV.trim_masks(e, f, S, Z)
            C = CV.camera_center(e)
            occ, hit = (walls.occluded(C, P, A2, B2) if (walls is not None and C is not None)
                        else (np.zeros(S.shape, bool), []))
            usable = tm["usable"] & ~occ
            a_px = 1.0 / (ppm * ppm)
            if usable.sum() * a_px < MIN_USABLE_M2:
                row["skipped"].append({"photo": e["photo"], "usable_m2": round(float(usable.sum() * a_px), 1),
                                       "occluded_m2": round(float((tm["usable"] & occ).sum() * a_px), 1)})
                continue
            img = CV._load_photo(e)
            if img is None:
                row["skipped"].append({"photo": e["photo"], "why": "写真が読めない"})
                continue
            warped = CV.warp_to_face(img, e, f, rect, ppm, tm["median"])
            warped[~usable] = GRAY
            ys, xs = np.nonzero(usable)
            pad = int(round(0.5 * ppm))
            x0, x1 = max(0, xs.min() - pad), min(usable.shape[1], xs.max() + 1 + pad)
            y0, y1 = max(0, ys.min() - pad), min(usable.shape[0], ys.max() + 1 + pad)
            stem = e["photo"]
            jp = out / f"{stem}.jpg"
            cv2.imencode(".jpg", warped[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(jp))
            cv2.imencode(".png", (usable[y0:y1, x0:x1] * 255).astype(np.uint8))[1].tofile(str(out / f"{stem}_mask.png"))
            meta = {"photo": stem, "face": fid, "kind": e.get("kind") or "plane", "image": jp.name, "mask": f"{stem}_mask.png",
                    "px_per_m": ppm, "crop_px": [int(x0), int(y0), int(x1), int(y1)],
                    "origin_m": {"s": rect[0] + x0 / ppm, "z_top": rect[3] - y0 / ppm},
                    "usable_m2": round(float(usable.sum() * a_px), 1),
                    "occluded_by_other_buildings_m2": round(float((tm["usable"] & occ).sum() * a_px), 1),
                    "occluded_by": hit, "median_photo_px_per_m": round(float(tm["median"]), 1),
                    "camera_xyz": [round(float(v), 2) for v in C] if C is not None else None,
                    "registration": e.get("check") or {}}
            (out / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            row["views"].append({k: meta[k] for k in ("photo", "kind", "usable_m2", "occluded_by_other_buildings_m2")})
        res["faces"].append(row)
    (root).mkdir(parents=True, exist_ok=True)
    (root / "views.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


PUBLIC_ROOT = Path("out") / "facade_public"


def public_views(bid: str, faces: list[str] | None = None, out_root: Path | None = None, registry: dict | None = None,
                 ppm: float | None = None) -> dict:
    """表示・動画・質感用の面の画像を、公開版の写真（掲示物・人名・顔・ナンバーを灰色の箱で塗った版。元の解像度の
    public_full_file があればそれ、無ければ長辺 1600 px の public_file）から作る。切り出しと使えない所は views
    （元の写真から作った 40 px/m の画像）と同じ範囲で、解像度だけ ppm（既定 PUBLIC_PX_PER_M＝80 px/m）にする。
    元の写真から作った画像は寸法を測るためだけに使い、人に見せる物はこちらにする（塗っていないため）。"""
    import cv2

    from .photos import load_index

    b = load_building(building_json(bid))
    reg = registry if registry is not None else load_registry()
    ents = {(e["photo"], e["face"]): e for e in expand_entries(entries_for(reg, bid), b)}
    idx = load_index()
    recs = {p["id"]: p for p in (idx["photos"] if isinstance(idx, dict) else idx)}
    src = (out_root or OUT_ROOT) / bid
    dst_root = PUBLIC_ROOT / bid
    res = {"building": bid, "faces": [], "missing_public": []}
    for fdir in sorted(p for p in src.iterdir() if p.is_dir()):
        fid = fdir.name
        if faces and fid not in faces:
            continue
        f = next((x for x in b["faces"] if x["id"] == fid), None)
        rect = face_rect(b, f) if f else None
        if rect is None:
            continue
        out = dst_root / fid
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for mj in sorted(fdir.glob("*.json")):
            if mj.name in RESULT_FILES:
                continue
            meta = json.loads(mj.read_text(encoding="utf-8"))
            if "image" not in meta:
                continue
            e = ents.get((meta["photo"], fid))
            rec = recs.get(meta["photo"])
            if e is None or not rec or not rec.get("public_file"):
                res["missing_public"].append(meta["photo"])
                continue
            # 元の解像度の公開版（photos publish --full、同じ箱を塗った物）があればそちらを使う（質感が約 3 倍細かくなる）
            full = rec.get("public_full_file")
            pub_rel = full if full and (Path("data") / "photos" / full).is_file() else rec["public_file"]
            pub0 = cv2.imdecode(np.fromfile(str(Path("data") / "photos" / pub_rel), np.uint8), cv2.IMREAD_COLOR)
            W, H = e["image_size"]
            pub = cv2.resize(pub0, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)     # 元の写真の画素の座標にそろえる
            p_out = float(ppm or PUBLIC_PX_PER_M)
            r = p_out / float(meta["px_per_m"])               # 窓の検出用（40 px/m）の切り出し → この解像度
            warped = CV.warp_to_face(pub, e, f, rect, p_out, meta["median_photo_px_per_m"])
            x0, y0, x1, y1 = (int(round(v * r)) for v in meta["crop_px"])
            crop = _crop_pad(warped, x0, y0, x1, y1, GRAY)
            mask = cv2.imdecode(np.fromfile(str(fdir / meta["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
            crop[~mask] = GRAY
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(out / meta["image"]))
            cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tofile(str(out / meta["mask"]))
            # 塗りつぶした所（個人情報の灰色の箱）のマスク：元の写真と公開版の差が大きい所。質感の合成でこの画素を使わない
            # （写真ごとに箱の位置が少しずつ違うので、そのまま重ねると箱と文字が透けて残る）
            orig = CV._load_photo(e)
            redact_name = None
            if orig is not None:
                o_small = cv2.resize(orig, (pub0.shape[1], pub0.shape[0]), interpolation=cv2.INTER_AREA)
                d = (np.abs(o_small.astype(np.int16) - pub0.astype(np.int16)).max(axis=2) > 48).astype(np.uint8)
                d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                d = cv2.dilate(cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((5, 5), np.uint8))
                dm = cv2.resize(d * 255, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
                wm = CV.warp_to_face(cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR), e, f, rect, p_out, meta["median_photo_px_per_m"])
                redact_name = f"{meta['photo']}_redact.png"
                cv2.imencode(".png", ((_crop_pad(wm, x0, y0, x1, y1, (0, 0, 0))[:, :, 0] > 127) * 255).astype(np.uint8))[1].tofile(str(out / redact_name))
            (out / mj.name).write_text(json.dumps({**meta, "px_per_m": p_out, "crop_px": [x0, y0, x1, y1],
                                                   "mask_from": f"out/facade/{bid}/{fid}/{meta['mask']}（{meta['px_per_m']:g} px/m を最近傍で拡大）",
                                                   "source": f"公開版の写真（data/photos/{pub_rel}、塗りつぶし済み）",
                                                   "public_long": int(max(pub0.shape[:2])),
                                                   **({"redact_mask": redact_name} if redact_name else {})},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
            n += 1
        res["faces"].append({"face": fid, "views": n})
    return res


def _crop_pad(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill) -> np.ndarray:
    """img[y0:y1, x0:x1] を切り出す。格子の端で四捨五入により 1 px 足りなければ fill で埋めて大きさをそろえる。"""
    out = np.empty((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    out[:] = fill
    c = img[max(0, y0):y1, max(0, x0):x1]
    out[max(0, -y0):max(0, -y0) + c.shape[0], max(0, -x0):max(0, -x0) + c.shape[1]] = c
    return out


def px_to_m(meta: dict, x: float, y: float) -> tuple[float, float]:
    """面の画像の px（切り出し後）→ 面の (s, z) m。"""
    ppm = meta["px_per_m"]
    return meta["origin_m"]["s"] + x / ppm, meta["origin_m"]["z_top"] - y / ppm


def m_to_px(meta: dict, s: float, z: float) -> tuple[float, float]:
    ppm = meta["px_per_m"]
    return (s - meta["origin_m"]["s"]) * ppm, (meta["origin_m"]["z_top"] - z) * ppm


def grid_image(img: np.ndarray, meta: dict, long_side: int = 2400) -> np.ndarray:
    """係（Claude）に見せる版：1 m ごとの細い線、5 m ごとの太い線と s・z の数字を描く。縮めても読めるように数字は大きめ。
    Claude は画素の座標ではなく、この目盛りで面の上の位置（m）を答える（縮めて見せても位置が狂わない）。"""
    import cv2

    h, w = img.shape[:2]
    k = min(1.0, long_side / max(h, w))
    out = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img.copy()
    ppm = meta["px_per_m"] * k
    s0, zt = meta["origin_m"]["s"], meta["origin_m"]["z_top"]
    H, W = out.shape[:2]
    s_end, z_end = s0 + W / ppm, zt - H / ppm
    for s in range(int(np.ceil(s0)), int(np.floor(s_end)) + 1):
        x = int(round((s - s0) * ppm))
        major = s % 5 == 0
        cv2.line(out, (x, 0), (x, H - 1), (0, 255, 255) if major else (200, 200, 200), 2 if major else 1)
        if major:
            cv2.putText(out, f"s{s}", (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for z in range(int(np.ceil(z_end)), int(np.floor(zt)) + 1):
        y = int(round((zt - z) * ppm))
        major = z % 5 == 0
        cv2.line(out, (0, y), (W - 1, y), (0, 255, 255) if major else (200, 200, 200), 2 if major else 1)
        if major:
            cv2.putText(out, f"z{z}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def grids(bid: str, faces: list[str] | None = None, out_root: Path | None = None) -> list[Path]:
    """各面の画像に目盛りを描いた版（<写真>_grid.jpg）を書く。"""
    import cv2

    root = (out_root or OUT_ROOT) / bid
    made = []
    for mj in sorted(root.glob("*/*.json")):
        if mj.name in RESULT_FILES or mj.parent.name.startswith("_"):
            continue
        if faces and mj.parent.name not in faces:
            continue
        meta = json.loads(mj.read_text(encoding="utf-8"))
        if "origin_m" not in meta:
            continue
        img = cv2.imdecode(np.fromfile(str(mj.parent / meta["image"]), np.uint8), cv2.IMREAD_COLOR)
        p = mj.parent / f"{meta['photo']}_grid.jpg"
        cv2.imencode(".jpg", grid_image(img, meta), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(p))
        made.append(p)
    return made


# ------------------------------------------------------------------ 窓の検出（写真 1 枚ずつ）

def _tiles(w: int, h: int, side: int = TILE_PX, overlap: float = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    def starts(n):
        if n <= side:
            return [0]
        step = int(side * (1 - overlap))
        xs = list(range(0, n - side, step)) + [n - side]
        return sorted(set(xs))
    return [(x, y, min(w, x + side), min(h, y + side)) for y in starts(h) for x in starts(w)]


def detect(bid: str, faces: list[str] | None = None, out_root: Path | None = None, prompt: str = "window .",
           box_threshold: float = 0.3, text_threshold: float = 0.25) -> dict:
    """各面の画像で窓の箱を拾い、面の m に直して <面>/detections.json に書く。"""
    import cv2
    from PIL import Image

    from .windows import MODEL_ID, MODEL_LICENSE, _infer, nms

    root = (out_root or OUT_ROOT) / bid
    res = {"building": bid, "model": MODEL_ID, "license": MODEL_LICENSE, "prompt": prompt,
           "box_threshold": box_threshold, "text_threshold": text_threshold, "faces": []}
    for fdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if faces and fdir.name not in faces:
            continue
        boxes_all = []
        for mj in sorted(fdir.glob("*.json")):
            if mj.name in RESULT_FILES:
                continue
            meta = json.loads(mj.read_text(encoding="utf-8"))
            if "image" not in meta or "origin_m" not in meta:
                continue
            img = cv2.imdecode(np.fromfile(str(fdir / meta["image"]), np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.fromfile(str(fdir / meta["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
            h, w = img.shape[:2]
            bx, sc = [], []
            for x0, y0, x1, y1 in _tiles(w, h):
                tile = Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                boxes, scores, _labels, _dev = _infer(tile, prompt, box_threshold, text_threshold)
                for (a, b_, c, d), s in zip(boxes, scores):
                    if (c - a) * (d - b_) > 0.25 * (x1 - x0) * (y1 - y0):
                        continue        # 切り出し全体を 1 つの窓と取った箱
                    bx.append([a + x0, b_ + y0, c + x0, d + y0])
                    sc.append(float(s))
            keep = nms(bx, sc) if bx else []
            ppm = meta["px_per_m"]
            for k in keep:
                a, b_, c, d = bx[k]
                cx, cy = int((a + c) / 2), int((b_ + d) / 2)
                if not (0 <= cy < h and 0 <= cx < w) or not mask[cy, cx]:
                    continue        # 箱の中心が使えない所（灰色）
                ww, hh = (c - a) / ppm, (d - b_) / ppm
                if not (BOX_LIMITS["w"][0] <= ww <= BOX_LIMITS["w"][1] and BOX_LIMITS["h"][0] <= hh <= BOX_LIMITS["h"][1]):
                    continue
                s0, z1 = px_to_m(meta, a, b_)
                s1, z0 = px_to_m(meta, c, d)
                boxes_all.append({"photo": meta["photo"], "score": round(sc[k], 3), "px": [round(v, 1) for v in (a, b_, c, d)],
                                  "s0": round(s0, 3), "s1": round(s1, 3), "z0": round(z0, 3), "z1": round(z1, 3)})
        doc = {"face": fdir.name, "boxes": boxes_all, "n_photos": len({x["photo"] for x in boxes_all})}
        (fdir / "detections.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        res["faces"].append({"face": fdir.name, "boxes": len(boxes_all), "photos": doc["n_photos"]})
    return res


# ------------------------------------------------------------------ 写真をまたいだ突き合わせ

def iou_m(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["s1"], b["s1"]) - max(a["s0"], b["s0"]))
    iz = max(0.0, min(a["z1"], b["z1"]) - max(a["z0"], b["z0"]))
    inter = ix * iz
    ua = (a["s1"] - a["s0"]) * (a["z1"] - a["z0"]) + (b["s1"] - b["s0"]) * (b["z1"] - b["z0"]) - inter
    return inter / ua if ua > 0 else 0.0


EDGES = ("s0", "s1", "z0", "z1")


def fuse_boxes(boxes: list[dict], match_iou: float = MATCH_IOU) -> list[dict]:
    """写真をまたいで同じ窓を束ねる。1 つの束には 1 枚の写真から 1 つの箱だけ。返り値は窓ごとの値・σ・根拠。"""
    tracks: list[dict] = []
    for bx in sorted(boxes, key=lambda x: -x["score"]):
        best, best_iou = None, match_iou
        for t in tracks:
            if bx["photo"] in t["photos"]:
                continue
            v = iou_m(bx, t["center"])
            if v >= best_iou:
                best, best_iou = t, v
        if best is None:
            best = {"members": [], "photos": set(), "center": dict(bx)}
            tracks.append(best)
        best["members"].append(bx)
        best["photos"].add(bx["photo"])
        best["center"] = {k: float(np.median([m[k] for m in best["members"]])) for k in EDGES}
    out = []
    for i, t in enumerate(sorted(tracks, key=lambda t: (-t["center"]["z1"], t["center"]["s0"]))):
        n = len(t["members"])
        win = {"id": f"w{i + 1:03d}", "n_photos": n}
        for k in EDGES:
            vals = np.array([m[k] for m in t["members"]], float)
            sd = float(np.std(vals, ddof=1)) if n >= 2 else None
            win[k] = round(float(np.median(vals)), 3)
            win[f"sd_{k}"] = round(sd, 3) if sd is not None else None
            win[f"sigma_{k}"] = round(float(np.sqrt(sd * sd / n + FLOOR_M ** 2)), 3) if sd is not None else None
        win["evidence"] = [{"photo": m["photo"], "px": m["px"], "score": m["score"]} for m in t["members"]]
        out.append(win)
    return out


def unusable_photos(bid: str, fid: str) -> dict[str, str]:
    """外壁の読み係が「使えない」とした写真（位置合わせが間の整数倍ずれている・切れ端など）と理由。
    data/reports/<棟>/gaiheki_<面>.json の photo_alignment の usable: false。"""
    p = Path("data") / "reports" / bid / f"gaiheki_{fid}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {a["photo"]: (a.get("note") or "") + (f"（ずれ {a['shift_m']} m）" if a.get("shift_m") else "")
            for a in d.get("photo_alignment") or [] if isinstance(a, dict) and a.get("usable") is False}


def fuse(bid: str, faces: list[str] | None = None, out_root: Path | None = None) -> dict:
    root = (out_root or OUT_ROOT) / bid
    res = {"building": bid, "floor_m": FLOOR_M, "match_iou": MATCH_IOU, "faces": []}
    for dj in sorted(root.glob("*/detections.json")):
        fid = dj.parent.name
        if faces and fid not in faces:
            continue
        det = json.loads(dj.read_text(encoding="utf-8"))
        bad = unusable_photos(bid, fid)
        boxes = [x for x in det["boxes"] if x["photo"] not in bad]
        wins = fuse_boxes(boxes)
        n_photos = len({m["photo"] for m in boxes})
        doc = {"face": fid, "made_at": datetime.now().isoformat(timespec="seconds"), "n_photos_with_boxes": n_photos,
               "excluded_photos": [{"photo": k, "why": f"外壁の読み係：{v}"} for k, v in sorted(bad.items())],
               "windows": wins,
               "summary": {"windows": len(wins), "seen_by_2_or_more": sum(1 for w in wins if w["n_photos"] >= 2),
                           "sigma_median_m": {k: _med([w[f"sigma_{k}"] for w in wins]) for k in EDGES}}}
        (dj.parent / "fused.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        res["faces"].append({"face": fid, **doc["summary"]})
    return res


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 3) if xs else None


# ------------------------------------------------------------------ 縁を壁の開口（凹み）の外形に寄せる

# LOD3 の窓は「壁の面に開いた穴を壁に投影した外形」。検出器の箱はガラスを囲むことが多く、凹んだ枠の上の陰（軒裏）や
# 下の縁台を外す。写真ごとに、箱の縁の近くで「明るい壁 → 暗い凹み」の変わり目を探し、写真どうしで突き合わせる。
# 探す範囲（m、箱の縁から外向きを正）：上は外へ 0.9 m・内へ 0.3 m、下は外へ 0.6 m・内へ 0.4 m、左右は外へ 0.5 m・内へ 0.3 m
SEARCH = {"z1": (0.9, 0.3), "z0": (0.6, 0.4), "s0": (0.5, 0.3), "s1": (0.5, 0.3)}
MIN_EDGE = 6.0           # 明るさの変わり目の強さの下限（0〜255 の明るさの、1 画素あたりの差を平滑化した値）


def _edge_1d(prof: np.ndarray, lo: int, hi: int, sign: int) -> tuple[int | None, float]:
    """prof の微分で、[lo, hi) の中の sign の向きに一番強い変わり目の位置と強さ。"""
    if hi - lo < 3:
        return None, 0.0
    g = np.gradient(prof)
    seg = sign * g[lo:hi]
    if not np.isfinite(seg).any():
        return None, 0.0
    i = int(np.nanargmax(seg))
    return lo + i, float(seg[i])


def refine_window(gray: np.ndarray, mask: np.ndarray, meta: dict, w: dict) -> dict:
    """1 枚の面の画像で、窓 w の 4 つの縁を凹みの外形に寄せる。見つからない縁は None。"""
    import cv2

    ppm = meta["px_per_m"]
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    g[~mask] = np.nan
    x0, y1 = m_to_px(meta, w["s0"], w["z0"])
    x1, y0 = m_to_px(meta, w["s1"], w["z1"])
    H, W = g.shape
    out = {}
    # 上下：箱の横幅の真ん中 70% の列の平均の明るさを、縦に見る
    cx0, cx1 = int(x0 + 0.15 * (x1 - x0)), int(x1 - 0.15 * (x1 - x0))
    if 0 <= cx0 < cx1 <= W:
        with np.errstate(all="ignore"):
            prof = np.nanmean(g[:, cx0:cx1], axis=1)
        a, b = SEARCH["z1"]
        y, s = _edge_1d(prof, max(0, int(y0 - a * ppm)), min(H, int(y0 + b * ppm)), -1)    # 下へ行くと暗くなる
        out["z1"] = (meta["origin_m"]["z_top"] - (y + 0.5) / ppm) if (y is not None and s >= MIN_EDGE) else None
        a, b = SEARCH["z0"]
        y, s = _edge_1d(prof, max(0, int(y1 - b * ppm)), min(H, int(y1 + a * ppm)), +1)    # 下へ行くと明るくなる
        out["z0"] = (meta["origin_m"]["z_top"] - (y + 0.5) / ppm) if (y is not None and s >= MIN_EDGE) else None
    # 左右：箱の高さの真ん中 70% の行の平均の明るさを、横に見る
    cy0, cy1 = int(y0 + 0.15 * (y1 - y0)), int(y1 - 0.15 * (y1 - y0))
    if 0 <= cy0 < cy1 <= H:
        with np.errstate(all="ignore"):
            prof = np.nanmean(g[cy0:cy1, :], axis=0)
        a, b = SEARCH["s0"]
        x, s = _edge_1d(prof, max(0, int(x0 - a * ppm)), min(W, int(x0 + b * ppm)), -1)    # 右へ行くと暗くなる
        out["s0"] = (meta["origin_m"]["s"] + (x + 0.5) / ppm) if (x is not None and s >= MIN_EDGE) else None
        a, b = SEARCH["s1"]
        x, s = _edge_1d(prof, max(0, int(x1 - b * ppm)), min(W, int(x1 + a * ppm)), +1)    # 右へ行くと明るくなる
        out["s1"] = (meta["origin_m"]["s"] + (x + 0.5) / ppm) if (x is not None and s >= MIN_EDGE) else None
    return out


def refine(bid: str, faces: list[str] | None = None, out_root: Path | None = None, min_n: int = 2) -> dict:
    """fused.json の窓（写真 min_n 枚以上）の縁を、写真ごとに凹みの外形へ寄せ、写真どうしで中央値と σ を出し直す。
    写真が縁を見つけられなかった縁は、検出器の縁のまま（refined に False）。refined.json に書く。"""
    import cv2

    root = (out_root or OUT_ROOT) / bid
    res = {"building": bid, "search_m": SEARCH, "min_edge": MIN_EDGE, "faces": []}
    for fj in sorted(root.glob("*/fused.json")):
        fdir = fj.parent
        if faces and fdir.name not in faces:
            continue
        wins = json.loads(fj.read_text(encoding="utf-8"))["windows"]
        cache: dict[str, tuple] = {}

        def view(photo):
            if photo not in cache:
                meta = json.loads((fdir / f"{photo}.json").read_text(encoding="utf-8"))
                img = cv2.imdecode(np.fromfile(str(fdir / meta["image"]), np.uint8), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imdecode(np.fromfile(str(fdir / meta["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
                cache[photo] = (img, mask, meta)
            return cache[photo]

        out = []
        for w in wins:
            if w["n_photos"] < min_n:
                continue
            per = []
            for e in w["evidence"]:
                img, mask, meta = view(e["photo"])
                per.append(refine_window(img, mask, meta, w))
            r = {"id": w["id"], "n_photos": w["n_photos"], "detector": {k: w[k] for k in EDGES}, "refined": {}}
            for k in EDGES:
                vals = [p[k] for p in per if p.get(k) is not None]
                if len(vals) >= 2:
                    sd = float(np.std(vals, ddof=1))
                    r[k] = round(float(np.median(vals)), 3)
                    r[f"sigma_{k}"] = round(float(np.sqrt(sd * sd / len(vals) + FLOOR_M ** 2)), 3)
                    r[f"n_{k}"] = len(vals)
                    r["refined"][k] = True
                else:
                    r[k] = w[k]
                    r[f"sigma_{k}"] = w.get(f"sigma_{k}")
                    r[f"n_{k}"] = w["n_photos"]
                    r["refined"][k] = False
            r["evidence"] = w["evidence"]
            out.append(r)
        doc = {"face": fdir.name, "made_at": datetime.now().isoformat(timespec="seconds"), "windows": out,
               "summary": {"windows": len(out), "edges_refined": {k: sum(1 for x in out if x["refined"][k]) for k in EDGES},
                           "sigma_median_m": {k: _med([x[f"sigma_{k}"] for x in out]) for k in EDGES}}}
        (fdir / "refined.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        res["faces"].append({"face": fdir.name, **doc["summary"]})
    return res


# ------------------------------------------------------------------ 並び方で整える（段の幅・列の線）

ROW_TOL_M = 0.6          # 窓の中心の高さがこれ以内なら同じ段
WIDTH_TOL = 0.2          # 段の幅の中央値からこの割合以内の窓を「よい窓」とする
LINE_TOL_M = 0.15        # 縁が、よい窓の縁の線からこれ以内なら、その線に乗っているとみなす


def regularize_windows(wins: list[dict]) -> tuple[list[dict], list[dict]]:
    """refined の窓を、段ごとの幅と列の線で整える。返り値（残した窓、捨てた窓と理由）。
    仮定（並び方）：同じ段の窓は同じ幅。窓の左右の縁は、どこかの段のよい窓の縁の線に乗る。これは外壁の読み係が確かめる。"""
    rows: list[list[dict]] = []
    for w in sorted(wins, key=lambda w: (w["z0"] + w["z1"]) / 2):
        zc = (w["z0"] + w["z1"]) / 2
        if rows and abs(zc - np.median([(x["z0"] + x["z1"]) / 2 for x in rows[-1]])) <= ROW_TOL_M:
            rows[-1].append(w)
        else:
            rows.append([w])
    kept, dropped = [], []
    good_all = []
    info = []
    for ri, row in enumerate(rows):
        both = [w for w in row if w["refined"].get("s0") and w["refined"].get("s1")] or row
        wmed = float(np.median([w["s1"] - w["s0"] for w in both]))
        good = [w for w in row if abs((w["s1"] - w["s0"]) - wmed) <= WIDTH_TOL * wmed]
        widths = [w["s1"] - w["s0"] for w in good]
        sw = float(np.std(widths, ddof=1) / np.sqrt(len(widths))) if len(widths) >= 2 else None
        info.append((ri, row, wmed, good, sw))
        good_all += good
    lines0 = [w["s0"] for w in good_all]
    lines1 = [w["s1"] - 0 for w in good_all]
    for ri, row, wmed, good, sw in info:
        cand = []
        for w in row:
            w = {**w, "row_index": ri, "row_width_m": round(wmed, 3)}
            if w in good or any(w["id"] == g["id"] for g in good):
                w["regularized"] = "そのまま（段の幅に合う）"
                cand.append(w)
                continue
            n0 = min((abs(w["s0"] - x) for x in lines0), default=9)
            n1 = min((abs(w["s1"] - x) for x in lines1), default=9)
            s_extra = sw if sw is not None else FLOOR_M
            if n0 <= LINE_TOL_M and n0 <= n1:
                w["s1"] = round(w["s0"] + wmed, 3)
                w["sigma_s1"] = round(float(np.hypot(w.get("sigma_s0") or FLOOR_M, s_extra)), 3)
                w["regularized"] = "右の縁を作り直した（左の縁＋段の幅）"
                cand.append(w)
            elif n1 <= LINE_TOL_M:
                w["s0"] = round(w["s1"] - wmed, 3)
                w["sigma_s0"] = round(float(np.hypot(w.get("sigma_s1") or FLOOR_M, s_extra)), 3)
                w["regularized"] = "左の縁を作り直した（右の縁−段の幅）"
                cand.append(w)
            else:
                dropped.append({"id": w["id"], "why": f"幅 {w['s1'] - w['s0']:.2f} m が段の幅 {wmed:.2f} m と合わず、縁が列の線に乗らない"})
        # 同じ段で 2 つ以上の窓にまたがる箱、ほかの窓に含まれる箱を捨てる（写真の枚数の多い方を残す）
        cand.sort(key=lambda w: -w["n_photos"])
        out_row: list[dict] = []
        for w in cand:
            area = (w["s1"] - w["s0"]) * (w["z1"] - w["z0"])
            ov = []
            for o in cand:
                if o is w:
                    continue
                ix = max(0.0, min(w["s1"], o["s1"]) - max(w["s0"], o["s0"]))
                iz = max(0.0, min(w["z1"], o["z1"]) - max(w["z0"], o["z0"]))
                if ix * iz > 0.2 * area:
                    ov.append(o)
            if len(ov) >= 2:
                dropped.append({"id": w["id"], "why": "同じ段の 2 つ以上の窓にまたがる"})
                continue
            dup = next((k for k in out_row if _inter_frac(w, k) > 0.6), None)
            if dup is not None:
                dropped.append({"id": w["id"], "why": f"{dup['id']} と同じ窓（重なり 6 割超）。写真の枚数の多い方を残した"})
                continue
            out_row.append(w)
        kept += out_row
    kept.sort(key=lambda w: (-w["z1"], w["s0"]))
    return kept, dropped


COL_MIN_ROWS = 2          # 列ごとの縁の探し直しに使う、同じ列の窓の段の数の下限
COL_SEARCH_M = (0.8, 0.5)  # 列の縁を探す範囲（外へ, 内へ）m


def _column_refine(wins: list[dict], fdir: Path) -> None:
    """同じ幅の窓が段をまたいで同じ列に並ぶとき（COL_MIN_ROWS 段以上）、その列の窓をまとめて、写真ごとに凹みの左右の縁
    （明るい壁 → 暗い凹み）を探し直し、写真どうしの中央値と σ にそろえる。仮定：同じ列の窓は左右の縁がそろう。"""
    import cv2

    cache: dict[str, tuple] = {}

    def view(photo):
        if photo not in cache:
            meta = json.loads((fdir / f"{photo}.json").read_text(encoding="utf-8"))
            img = cv2.imdecode(np.fromfile(str(fdir / meta["image"]), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imdecode(np.fromfile(str(fdir / meta["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
            g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
            g[~mask] = np.nan
            cache[photo] = (g, meta)
        return cache[photo]

    used = set()
    columns: list[dict] = []          # 探し直せた列（左右の縁・σ・幅）。まとめきれなかった窓をそろえるのに使う
    for w in sorted(wins, key=lambda x: -x["n_photos"]):
        if id(w) in used:
            continue
        wd = w["s1"] - w["s0"]
        c = (w["s0"] + w["s1"]) / 2
        col = [x for x in wins if id(x) not in used and abs((x["s1"] - x["s0"]) - wd) <= WIDTH_TOL * wd
               and abs((x["s0"] + x["s1"]) / 2 - c) <= 0.45 * wd]
        if len({x.get("row_index") for x in col}) < COL_MIN_ROWS:
            continue
        used.update(id(x) for x in col)
        columns.append({"members": col})
        s0m = float(np.median([x["s0"] for x in col]))
        s1m = float(np.median([x["s1"] for x in col]))
        photos = sorted({e["photo"] for x in col for e in x["evidence"]})
        found = {"s0": [], "s1": []}
        for ph in photos:
            g, meta = view(ph)
            ppm = meta["px_per_m"]
            H, W = g.shape
            profs = []
            for x in col:
                ya = int(m_to_px(meta, 0, x["z1"])[1] + 0.15 * (x["z1"] - x["z0"]) * ppm)
                yb = int(m_to_px(meta, 0, x["z0"])[1] - 0.15 * (x["z1"] - x["z0"]) * ppm)
                ya, yb = max(0, ya), min(H, yb)
                if yb - ya < 3:
                    continue
                with np.errstate(all="ignore"):
                    profs.append(np.nanmean(g[ya:yb, :], axis=0))
            if not profs:
                continue
            with np.errstate(all="ignore"):
                prof = np.nanmean(np.vstack(profs), axis=0)
            out_, in_ = COL_SEARCH_M
            xa, xb = int(m_to_px(meta, s0m - out_, 0)[0]), int(m_to_px(meta, s0m + in_, 0)[0])
            xx, st = _edge_1d(prof, max(0, xa), min(W, xb), -1)
            if xx is not None and st >= MIN_EDGE and xa >= 0 and xb <= W:
                found["s0"].append(meta["origin_m"]["s"] + (xx + 0.5) / ppm)
            xa, xb = int(m_to_px(meta, s1m - in_, 0)[0]), int(m_to_px(meta, s1m + out_, 0)[0])
            xx, st = _edge_1d(prof, max(0, xa), min(W, xb), +1)
            if xx is not None and st >= MIN_EDGE and xa >= 0 and xb <= W:
                found["s1"].append(meta["origin_m"]["s"] + (xx + 0.5) / ppm)
        for k in ("s0", "s1"):
            vals = found[k]
            if len(vals) < 2:
                continue
            med = float(np.median(vals))
            sd = float(np.std(vals, ddof=1))
            sig = float(np.sqrt(sd * sd / len(vals) + FLOOR_M ** 2))
            for x in col:
                x[k] = round(med, 3)
                x[f"sigma_{k}"] = round(sig, 3)
                x[f"n_{k}"] = len(vals)
                x["regularized"] = (x.get("regularized", "") + f"／{k} は同じ列の {len(col)} 窓をまとめて写真 {len(vals)} 枚で探し直した").lstrip("／")
            columns[-1][k] = (round(med, 3), round(sig, 3))
    # まとめきれなかった窓（幅が少し違う・半分ずれた箱）を、近い列の縁にそろえる
    for x in wins:
        if id(x) in used:
            continue
        cx, wx = (x["s0"] + x["s1"]) / 2, x["s1"] - x["s0"]
        best = None
        for c in columns:
            if "s0" not in c or "s1" not in c:
                continue
            s0, s1 = c["s0"][0], c["s1"][0]
            wd, cc = s1 - s0, (s0 + s1) / 2
            if abs(wx - wd) <= 0.35 * wd and abs(cx - cc) <= 0.6 * wd and (best is None or abs(cx - cc) < best[0]):
                best = (abs(cx - cc), c)
        if best is not None:
            c = best[1]
            for k in ("s0", "s1"):
                x[k], x[f"sigma_{k}"] = c[k]
            x["regularized"] = (x.get("regularized", "") + "／左右の縁を近い列の縁にそろえた").lstrip("／")


def _dedupe_after_columns(wins: list[dict]) -> tuple[list[dict], list[dict]]:
    """列の縁にそろえた後、同じ段で 3 割以上重なる窓を 1 つにする（写真の枚数の多い方を残す）。"""
    kept, dropped = [], []
    for w in sorted(wins, key=lambda w: -w["n_photos"]):
        dup = next((k for k in kept if k.get("row_index") == w.get("row_index") and iou_m(w, k) > 0.3), None)
        if dup is not None:
            dropped.append({"id": w["id"], "why": f"列の縁にそろえた後 {dup['id']} と同じ窓になった"})
            continue
        kept.append(w)
    kept.sort(key=lambda w: (-w["z1"], w["s0"]))
    return kept, dropped


def _inter_frac(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["s1"], b["s1"]) - max(a["s0"], b["s0"]))
    iz = max(0.0, min(a["z1"], b["z1"]) - max(a["z0"], b["z0"]))
    small = min((a["s1"] - a["s0"]) * (a["z1"] - a["z0"]), (b["s1"] - b["s0"]) * (b["z1"] - b["z0"]))
    return ix * iz / small if small > 0 else 0.0


def regularize(bid: str, faces: list[str] | None = None, out_root: Path | None = None) -> dict:
    root = (out_root or OUT_ROOT) / bid
    res = {"building": bid, "faces": []}
    for rj in sorted(root.glob("*/refined.json")):
        if faces and rj.parent.name not in faces:
            continue
        wins = json.loads(rj.read_text(encoding="utf-8"))["windows"]
        kept, dropped = regularize_windows(wins)
        _column_refine(kept, rj.parent)
        kept, dropped2 = _dedupe_after_columns(kept)
        dropped += dropped2
        doc = {"face": rj.parent.name, "made_at": datetime.now().isoformat(timespec="seconds"),
               "assumption": "同じ段の窓は同じ幅。窓の左右の縁は、どこかの段のよい窓の縁の線に乗る。同じ列の窓は左右の縁がそろう（外壁の読み係が確かめる）",
               "windows": kept, "dropped": dropped,
               "summary": {"windows": len(kept), "dropped": len(dropped),
                           "adjusted": sum(1 for w in kept if not w["regularized"].startswith("そのまま"))}}
        (rj.parent / "final.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        res["faces"].append({"face": rj.parent.name, **doc["summary"]})
    return res


# ------------------------------------------------------------------ 棟 JSON の窓と比べる

def compare(bid: str, faces: list[str] | None = None, out_root: Path | None = None, min_n: int = 2) -> dict:
    """写真から出した窓（n_photos ≥ min_n）と、棟 JSON の窓を面の m で突き合わせる。"""
    from .windows import face_window_rects

    b = load_building(building_json(bid))
    root = (out_root or OUT_ROOT) / bid
    res = {"building": bid, "min_n": min_n, "faces": []}
    for fdir in sorted(p for p in root.iterdir() if p.is_dir()):
        fj = next((q for q in (fdir / "final.json", fdir / "fused.json") if q.exists()), None)
        fid = fdir.name
        if fj is None or (faces and fid not in faces):
            continue
        f = next((x for x in b["faces"] if x["id"] == fid), None)
        if f is None:
            continue
        model = face_window_rects(b, f)
        seen = [w for w in json.loads(fj.read_text(encoding="utf-8"))["windows"] if w["n_photos"] >= min_n]
        pairs, used = [], set()
        for w in sorted(seen, key=lambda w: -w["n_photos"]):
            best, bi = None, 0.3
            for j, m in enumerate(model):
                if j in used:
                    continue
                v = iou_m(w, m)
                if v >= bi:
                    best, bi = j, v
            if best is not None:
                used.add(best)
                m = model[best]
                d = {k: round(w[k] - m[k], 3) for k in EDGES}
                z = {k: (round(d[k] / w[f"sigma_{k}"], 2) if w.get(f"sigma_{k}") else None) for k in EDGES}
                pairs.append({"photo_window": w["id"], "model": {"row": m["row"], "bay": m["bay"], "k": m["k"]},
                              "iou": round(bi, 3), "diff_m": d, "diff_over_sigma": z})
        only_photo = [w["id"] for w in seen if w["id"] not in {p["photo_window"] for p in pairs}]
        only_model = [{"row": m["row"], "bay": m["bay"], "k": m["k"]} for j, m in enumerate(model) if j not in used]
        absd = {k: _med([abs(p["diff_m"][k]) for p in pairs]) for k in EDGES}
        doc = {"face": fid, "model_windows": len(model), "photo_windows": len(seen), "matched": len(pairs),
               "only_in_photos": only_photo, "only_in_model": len(only_model),
               "median_abs_diff_m": absd, "pairs": pairs}
        (fj.parent / "compare.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        res["faces"].append({k: doc[k] for k in ("face", "model_windows", "photo_windows", "matched", "median_abs_diff_m")}
                            | {"only_in_photos": len(only_photo), "only_in_model": len(only_model)})
    return res


# ------------------------------------------------------------------ コマンド

def add_args(p) -> None:
    p.add_argument("action", choices=["views", "detect", "fuse", "refine", "regularize", "compare", "public", "all"])
    p.add_argument("building")
    p.add_argument("--faces", default=None, help="面 id（カンマ区切り。省略で写真のある面全部）")
    p.add_argument("--no-occlusion", action="store_true", help="PLATEAU の他の棟による遮蔽を使わない")
    p.add_argument("--min-n", type=int, default=2, help="compare で比べる窓の、写った写真の枚数の下限")
    p.add_argument("--px-per-m", type=float, default=None,
                   help=f"views の解像度（既定 {PX_PER_M:g}。上げた時は窓の検出用と混ざらないよう out/facade_ppm<N> に書く）。"
                        f"public では公開版の面の画像の解像度（既定 {PUBLIC_PX_PER_M:g}）")
    p.add_argument("--photos", default=None, help="views で使う写真 id（カンマ区切り。省略で全部）")


def run_cli(a) -> int:
    faces = [x.strip() for x in a.faces.split(",")] if a.faces else None
    steps = ["views", "detect", "fuse", "refine", "regularize", "compare"] if a.action == "all" else [a.action]
    for st in steps:
        if st == "views":
            ppm = float(getattr(a, "px_per_m", PX_PER_M) or PX_PER_M)
            photos = [x.strip() for x in a.photos.split(",")] if getattr(a, "photos", None) else None
            if (ppm != PX_PER_M or photos) and a.action == "all":
                print("--px-per-m・--photos は views だけで使います（all では窓の検出用の 40 px/m・全部の写真で作る）")
                return 2
            # 解像度を変えた時・写真を絞った時は、窓の検出が読む out/facade（と views.json）を上書きしない
            root = Path("out") / f"facade_ppm{ppm:g}" if (ppm != PX_PER_M or photos) else None
            r = views(a.building, faces, ppm=ppm, out_root=root, use_occlusion=not a.no_occlusion, photos=photos)
            if root:
                print(f"解像度 {ppm:g} px/m の面の画像 → {root / a.building}")
            for f in r["faces"]:
                occ = sum(v["occluded_by_other_buildings_m2"] for v in f["views"])
                print(f"{a.building} {f['face']}: 面の画像 {len(f['views'])} 枚（他の棟の陰で外した計 {occ:.0f} m²、"
                      f"使える所が少なく作らなかった写真 {len(f['skipped'])} 枚）")
        elif st == "detect":
            r = detect(a.building, faces)
            for f in r["faces"]:
                print(f"{a.building} {f['face']}: 窓の箱 {f['boxes']}（写真 {f['photos']} 枚）")
        elif st == "fuse":
            r = fuse(a.building, faces)
            for f in r["faces"]:
                print(f"{a.building} {f['face']}: 窓 {f['windows']}（2 枚以上で照合 {f['seen_by_2_or_more']}）σ の中央値 {f['sigma_median_m']}")
        elif st == "refine":
            r = refine(a.building, faces)
            for f in r["faces"]:
                print(f"{a.building} {f['face']}: 縁を凹みの外形に寄せた {f['edges_refined']}（窓 {f['windows']}）σ の中央値 {f['sigma_median_m']}")
        elif st == "regularize":
            r = regularize(a.building, faces)
            for f in r["faces"]:
                print(f"{a.building} {f['face']}: 並び方で整えた窓 {f['windows']}（直した {f['adjusted']}・捨てた {f['dropped']}）")
        elif st == "public":
            p_pub = float(getattr(a, "px_per_m", None) or PUBLIC_PX_PER_M)
            r = public_views(a.building, faces, ppm=p_pub)
            for f in r["faces"]:
                print(f"{a.building} {f['face']}: 公開版の面の画像 {f['views']} 枚（{p_pub:g} px/m、out/facade_public、塗りつぶしのマスク付き）")
            if r["missing_public"]:
                print(f"公開版の無い写真 {len(r['missing_public'])} 枚")
        elif st == "compare":
            r = compare(a.building, faces, min_n=a.min_n)
            for f in r["faces"]:
                print(f"{a.building} {f['face']}: 模型の窓 {f['model_windows']}・写真の窓 {f['photo_windows']}・対応 {f['matched']}"
                      f"（写真だけ {f['only_in_photos']}・模型だけ {f['only_in_model']}）ずれの中央値 {f['median_abs_diff_m']}")
    return 0

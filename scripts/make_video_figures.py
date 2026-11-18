"""動画・スライド用の見せ場の画像を、公開版の写真（掲示物・人名・顔を塗った版）から作る（out/video/）。

1. fig_windows_honbu_f6.jpg：本部棟 f6 の面の画像に、棟の記録の窓（青）と写真から出した窓（赤）を重ねる
2. fig_misaligned_honbu_f6.jpg：基準の写真と、位置合わせが 2 間（10.6 m）ずれていた写真を、面の座標で上下に並べる
   （外壁の読み係が、くり返さない物＝垂れ幕の位置の違いで見抜いた例。垂れ幕は「掲示物」の灰色の箱で塗ってある）

使い方：.venv/Scripts/python -I -X utf8 scripts/make_video_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recorder import facade  # noqa: E402
from recorder.schema import load_building  # noqa: E402
from recorder.windows import face_window_rects  # noqa: E402

OUT = ROOT / "out" / "video"
PUB = ROOT / "out" / "facade_public"
FONT = next((p for p in (r"C:\Windows\Fonts\YuGothB.ttc", r"C:\Windows\Fonts\meiryob.ttc", r"C:\Windows\Fonts\msgothic.ttc")
             if Path(p).exists()), None)


def font(size):
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def view(bid, fid, photo):
    d = PUB / bid / fid
    meta = json.loads((d / f"{photo}.json").read_text(encoding="utf-8"))
    img = cv2.imdecode(np.fromfile(str(d / meta["image"]), np.uint8), cv2.IMREAD_COLOR)
    return img, meta


def caption(img_bgr, lines, h=None):
    """画像の上に帯をつけて説明の文字を書く（日本語の字形のため PIL）。"""
    W = img_bgr.shape[1]
    size = max(20, W // 55)
    h = h or int(size * 1.6 * len(lines) + size)
    band = Image.new("RGB", (W, h), (255, 255, 255))
    dr = ImageDraw.Draw(band)
    for i, (t, col) in enumerate(lines):
        dr.text((size // 2, size // 2 + int(i * size * 1.6)), t, fill=col, font=font(size))
    return np.vstack([cv2.cvtColor(np.array(band), cv2.COLOR_RGB2BGR), img_bgr])


def rect(img, meta, r, col, th):
    x0, y0 = facade.m_to_px(meta, r["s0"], r["z1"])
    x1, y1 = facade.m_to_px(meta, r["s1"], r["z0"])
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, th)


def fig_windows():
    bid, fid, photo = "honbu", "f6", "PXL_20261103_042900775"
    img, meta = view(bid, fid, photo)
    b = load_building(ROOT / "data" / "buildings" / f"{bid}.json")
    f = next(x for x in b["faces"] if x["id"] == fid)
    for r in face_window_rects(b, f):
        rect(img, meta, r, (220, 120, 20), 3)
    fin = json.loads((ROOT / "out" / "facade" / bid / fid / "final.json").read_text(encoding="utf-8"))["windows"]
    for w in fin:
        rect(img, meta, w, (40, 40, 230), 2)
    # この写真が写している範囲（s0〜33 m）だけを使う。右端は面の画像が灰色で、他の写真の窓を重ねても確かめられない
    x_end = int(facade.m_to_px(meta, 33.0, 0)[0])
    img = img[:, :max(1, min(img.shape[1], x_end))]
    img = caption(img, [("本部棟の正面（f6）：写真 1 枚を面に正面化した画像に、窓を重ねた", (20, 20, 20)),
                        ("青＝これまでの記録の窓（ガラスだけを窓にしていて下端が 0.65 m 高い）", (20, 110, 220)),
                        ("赤＝写真 18 枚から検出し、写真どうしで照合した窓（凹みの外形。左右・上端は青と 5 cm 以内で一致）", (220, 30, 30))])
    return img


def fig_misaligned():
    bid, fid = "honbu", "f6"
    ref, mref = view(bid, fid, "PXL_20261103_042900775")
    bad, mbad = view(bid, fid, "PXL_20261103_042830256")
    ppm = mref["px_per_m"]
    s_lo, s_hi = 0.0, 40.0
    z_top = max(mref["origin_m"]["z_top"], mbad["origin_m"]["z_top"])
    z_lo = min(mref["origin_m"]["z_top"] - ref.shape[0] / ppm, mbad["origin_m"]["z_top"] - bad.shape[0] / ppm)
    H, W = int((z_top - z_lo) * ppm), int((s_hi - s_lo) * ppm)

    def place(img, meta):
        canvas = np.full((H, W, 3), 235, np.uint8)
        x0 = int(round((meta["origin_m"]["s"] - s_lo) * ppm))
        y0 = int(round((z_top - meta["origin_m"]["z_top"]) * ppm))
        h, w = img.shape[:2]
        xs, ys = max(0, x0), max(0, y0)
        xe, ye = min(W, x0 + w), min(H, y0 + h)
        if xe > xs and ye > ys:
            canvas[ys:ye, xs:xe] = img[ys - y0:ye - y0, xs - x0:xe - x0]
        for s in range(int(s_lo), int(s_hi) + 1, 5):          # 5 m ごとの目盛り
            x = int((s - s_lo) * ppm)
            cv2.line(canvas, (x, 0), (x, H - 1), (0, 200, 255), 2)
            cv2.putText(canvas, f"s{s}", (x + 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return canvas

    a, b = place(ref, mref), place(bad, mbad)
    # 垂れ幕（灰色の箱）の位置を矢印で示す：基準 s17.6〜25.8、ずれた写真 s7.0〜15.3（外壁の読み係の読み）
    for img, (s0, s1) in ((a, (17.6, 25.8)), (b, (7.0, 15.3))):
        y = H - 40
        cv2.arrowedLine(img, (int((s0 - s_lo) * ppm), y), (int((s1 - s_lo) * ppm), y), (40, 40, 230), 4, tipLength=0.03)
        cv2.arrowedLine(img, (int((s1 - s_lo) * ppm), y), (int((s0 - s_lo) * ppm), y), (40, 40, 230), 4, tipLength=0.03)
    a = caption(a, [("基準の写真（042900775）：垂れ幕（灰色に塗った掲示物）は s17.6〜25.8", (20, 20, 20))])
    b = caption(b, [("別の写真（042830256）：同じ垂れ幕が s7.0〜15.3 に来ている → 位置合わせが 2 間（10.6 m）ずれていた", (220, 30, 30)),
                    ("窓の並びはくり返すので検出器では気づけない。Claude（外壁の読み係）が、くり返さない物の位置で見抜いた", (20, 20, 20))])
    gap = np.full((24, W, 3), 255, np.uint8)
    return np.vstack([a, gap, b])


def fig_sheet(cols: int = 3, scale_ppm: float = 8.0):
    """本部棟 f6 の写真 1 枚ずつの面の画像（公開版）を、面の座標の同じ位置に置いて並べる。位置合わせのずれた写真は除く。"""
    bid, fid = "honbu", "f6"
    d = PUB / bid / fid
    bad = facade.unusable_photos(bid, fid)
    metas = []
    for mj in sorted(d.glob("*.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if "image" in m and m["photo"] not in bad and m["usable_m2"] >= 150:
            metas.append(m)
    metas.sort(key=lambda m: -m["usable_m2"])
    metas = metas[:9]
    s_hi = 50.0
    z_top = max(m["origin_m"]["z_top"] for m in metas)
    z_lo = min(m["origin_m"]["z_top"] - m["crop_px"][3] / m["px_per_m"] + m["crop_px"][1] / m["px_per_m"] for m in metas)
    W, H = int(s_hi * scale_ppm), int((z_top - z_lo) * scale_ppm)
    tiles = []
    for m in metas:
        img = cv2.imdecode(np.fromfile(str(d / m["image"]), np.uint8), cv2.IMREAD_COLOR)
        k = scale_ppm / m["px_per_m"]
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        canvas = np.full((H, W, 3), 240, np.uint8)
        x0 = int(round(m["origin_m"]["s"] * scale_ppm))
        y0 = int(round((z_top - m["origin_m"]["z_top"]) * scale_ppm))
        h, w = small.shape[:2]
        xs, ys, xe, ye = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xe > xs and ye > ys:
            canvas[ys:ye, xs:xe] = small[ys - y0:ye - y0, xs - x0:xe - x0]
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (200, 200, 200), 1)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full((H, W, 3), 255, np.uint8))
    gap_v = np.full((H, 8, 3), 255, np.uint8)
    rows = [np.hstack(sum([[t, gap_v] for t in tiles[i:i + cols]], [])[:-1]) for i in range(0, len(tiles), cols)]
    gap_h = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.vstack(sum([[r, gap_h] for r in rows], [])[:-1])
    return caption(grid, [("写真 1 枚ずつを面の座標に正面化（貼り合わせない）。どの写真も面の上の同じ位置に並ぶ", (20, 20, 20)),
                          ("灰色は使えない所（面の外・斜め・他の棟の陰）。掲示物・人名・顔は灰色の箱で塗った公開版", (90, 90, 90))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig_windows_honbu_f6.jpg", fig_windows), ("fig_misaligned_honbu_f6.jpg", fig_misaligned),
                     ("fig_sheet_honbu_f6.jpg", fig_sheet)):
        img = fn()
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / name))
        print(OUT / name, img.shape)


if __name__ == "__main__":
    main()

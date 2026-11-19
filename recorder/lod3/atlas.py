"""質感の画像を 1 つの CityGML の分ずつ、大きな画像（アトラス）に詰める（11/19）。

PLATEAU 標準製品仕様書 9.4 節と作業手順書の決まり（`D:\\PLATEAU\\lod3\\research\\texture_prior_art.md` の 3-1）：
- 複数の地物の質感を 1 枚にまとめる（アトラス化）のを推奨。まとめる上限は 1 つの CityGML ファイルの分。
- 1 辺は 2 の累乗で 2048 px が基本、大規模建築物は 4096 px まで。超える分は画像を分けてよい。
- 元の画像を回転・拡縮せずに矩形のまま詰め、テクスチャ座標を詰めた後の座標に置き換える。詰める起点は左上。
- 面ごとに切り出す時、外側に 2 px ほどの余白を含める（ここでは縁の画素をくり返して余白にする。継ぎ目のにじみを防ぐ）。
- 背景は黒か灰色 (90,90,90)。

詰め方は「高さの順に棚に並べる」簡単な方法（first-fit decreasing height）。高い物から順に、入る棚の左から置く。
"""
from __future__ import annotations

import numpy as np

PAGE = 4096                 # 1 枚の画像の 1 辺の上限
MARGIN = 2                  # 面の外側の余白（px）
PAD_RGB = (90, 90, 90)


def _pow2(n: int) -> int:
    return 1 << (max(1, int(n)) - 1).bit_length()


class Atlas:
    """add() で画像を足し、pack() で詰め、page_image() で書き出す。uv() で画像の中の画素の位置 → アトラスの (u, v)。"""

    def __init__(self, page: int = PAGE, margin: int = MARGIN):
        self.page, self.margin = page, margin
        self.items: list[np.ndarray] = []
        self.place: list[tuple[int, int, int]] = []     # (何枚目, 左, 上)：余白を含めた箱の左上
        self.sizes: list[tuple[int, int]] = []          # 何枚目の画像の (幅, 高さ)

    def add(self, img: np.ndarray) -> int:
        h, w = img.shape[:2]
        if max(w, h) + 2 * self.margin > self.page:
            raise ValueError(f"画像 {w}×{h} px が 1 辺 {self.page} px（余白込み）に入らない")
        self.items.append(img)
        return len(self.items) - 1

    def pack(self) -> None:
        m, P = self.margin, self.page
        order = sorted(range(len(self.items)), key=lambda i: (-self.items[i].shape[0], -self.items[i].shape[1]))
        shelves: list[list[int]] = []                   # [何枚目, 上, 高さ, 使った幅]
        used_h: list[int] = []                          # 何枚目ごとの、棚で使った高さ
        self.place = [(0, 0, 0)] * len(self.items)
        for i in order:
            h, w = self.items[i].shape[0] + 2 * m, self.items[i].shape[1] + 2 * m
            sh = next((s for s in shelves if s[2] >= h and P - s[3] >= w), None)
            if sh is None:
                pg = next((k for k, uh in enumerate(used_h) if P - uh >= h), None)
                if pg is None:
                    used_h.append(0)
                    pg = len(used_h) - 1
                sh = [pg, used_h[pg], h, 0]
                used_h[pg] += h
                shelves.append(sh)
            self.place[i] = (sh[0], sh[3], sh[1])
            sh[3] += w
        self.sizes = []
        for pg in range(len(used_h)):
            ws = [self.place[i][1] + self.items[i].shape[1] + 2 * m for i in range(len(self.items)) if self.place[i][0] == pg]
            hs = [self.place[i][2] + self.items[i].shape[0] + 2 * m for i in range(len(self.items)) if self.place[i][0] == pg]
            self.sizes.append((min(P, _pow2(max(ws))), min(P, _pow2(max(hs)))))

    @property
    def n_pages(self) -> int:
        return len(self.sizes)

    def page_image(self, pg: int) -> np.ndarray:
        import cv2

        W, H = self.sizes[pg]
        out = np.empty((H, W, 3), np.uint8)
        out[:] = PAD_RGB[::-1]
        m = self.margin
        for i, img in enumerate(self.items):
            p, x, y = self.place[i]
            if p != pg:
                continue
            box = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REPLICATE)
            out[y:y + box.shape[0], x:x + box.shape[1]] = box
        return out

    def rect(self, i: int) -> dict:
        """画像 i がアトラスのどこにあるか（余白を除いた中身の左上と大きさ、px）。"""
        p, x, y = self.place[i]
        h, w = self.items[i].shape[:2]
        return {"page": p, "x": x + self.margin, "y": y + self.margin, "w": w, "h": h}

    def uv(self, i: int, xs, ys) -> list[tuple[float, float]]:
        """画像 i の中の位置（左上の角が 0、右・下へ px。画素の中心ではなく縁が整数）→ アトラスの (u, v)。v は下から。"""
        p, x, y = self.place[i]
        W, H = self.sizes[p]
        return [((x + self.margin + float(a)) / W, 1.0 - (y + self.margin + float(b)) / H) for a, b in zip(xs, ys)]

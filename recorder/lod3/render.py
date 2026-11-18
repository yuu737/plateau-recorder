"""書き出した CityGML を読み直して、地物の種類ごとに色を付けた絵を描く（変換の途中の値は使わない）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from lxml import etree  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from pyproj import Transformer  # noqa: E402

B = "{http://www.opengis.net/citygml/building/2.0}"
GM = "{http://www.opengis.net/gml}"
COL = {"WallSurface": "#d9d4c7", "RoofSurface": "#c0504d", "GroundSurface": "#555555", "OuterCeilingSurface": "#8064a2",
       "OuterFloorSurface": "#4bacc6", "Window": "#4f81bd", "Door": "#f79646", "BuildingInstallation": "#9bbb59"}
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]


def rings_of(el):
    out = []
    for pol in el.iter(GM + "Polygon"):
        rs = []
        for lr in pol.iter(GM + "LinearRing"):
            v = np.array(lr.find(GM + "posList").text.split(), float).reshape(-1, 3)
            rs.append(v)
        out.append(rs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recorder lod3 render")
    ap.add_argument("gml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--elev", type=float, default=25)
    ap.add_argument("--azim", type=float, default=-120)
    args = ap.parse_args(argv)
    tr = Transformer.from_crs("EPSG:6668", "EPSG:6675", always_xy=True)
    tree = etree.parse(args.gml, etree.XMLParser(huge_tree=True))
    blds = list(tree.getroot().iter(B + "Building"))
    fig = plt.figure(figsize=(7 * len(blds), 7))
    for k, bel in enumerate(blds):
        ax = fig.add_subplot(1, len(blds), k + 1, projection="3d")
        polys, cols = [], []
        counts = {}
        for kind in ("WallSurface", "RoofSurface", "GroundSurface", "OuterCeilingSurface", "OuterFloorSurface", "BuildingInstallation"):
            for el in bel.iter(B + kind):
                for rs in rings_of(el if kind == "BuildingInstallation" else el.find(B + "lod3MultiSurface")):
                    polys.append(rs[0])
                    cols.append(COL[kind])
                counts[kind] = counts.get(kind, 0) + 1
        for kind in ("Window", "Door"):
            for el in bel.iter(B + kind):
                for rs in rings_of(el):
                    polys.append(rs[0])
                    cols.append(COL[kind])
                counts[kind] = counts.get(kind, 0) + 1
        allp = np.vstack(polys)
        x, y = tr.transform(allp[:, 1], allp[:, 0])
        o = np.array([np.mean(x), np.mean(y), allp[:, 2].min()])
        loc = []
        for P in polys:
            px, py = tr.transform(P[:, 1], P[:, 0])
            loc.append(np.column_stack([px - o[0], py - o[1], P[:, 2] - o[2]]))
        pc = Poly3DCollection(loc, facecolors=cols, edgecolors="#333333", linewidths=0.2)
        ax.add_collection3d(pc)
        L = np.vstack(loc)
        r = (L.max(0) - L.min(0)).max() / 2
        c = (L.max(0) + L.min(0)) / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(0, 2 * r)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        name = bel.get(GM + "id")
        ax.set_title(f"{name[:13]}…\n" + " ".join(f"{kk.replace('Surface', '')}:{v}" for kk, v in counts.items()), fontsize=8)
    out = args.out or str(Path(args.gml).with_suffix(".png"))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    sys.exit(main())

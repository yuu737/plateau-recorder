"""PLATEAU 仕様（CityGML 2.0＋i-UR 3.1）の LOD3 を書き出し・検査する。

- build：棟 JSON → LOD3（PLATEAU の元の LOD0/LOD1 と属性を写して LOD3 を足す）
- quality：値ごとの確かさ（σ）を地図情報レベルの等級に、出どころを PLATEAU のコードに
- gmlcheck：XML スキーマと立体の閉じの検査
- render：書き出した GML を読み直して描く
- aerial：国土地理院の航空写真（屋根の倒れ込みの照合）
- schemas：CityGML・i-UR の XSD を手元に写す

使う資料は data/plateau/（XSD・コードリスト・名古屋 2022 の 52365768。GML と航空写真の写しは git 管理外）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLATEAU_DIR = ROOT / "data" / "plateau"
XSD_DIR = PLATEAU_DIR / "xsd"
CODELISTS_DIR = PLATEAU_DIR / "codelists"
SOURCE_GML = PLATEAU_DIR / "udx" / "bldg" / "52365768_bldg_6697_op.gml"
OUT_DIR = ROOT / "out" / "lod3"

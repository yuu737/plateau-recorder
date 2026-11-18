#!/usr/bin/env bash
# 書き出した LOD3 を、PLATEAU 公式の変換ツール（PLATEAU GIS Converter のコマンド版 nusamai）で変換できるか確かめる。
# WSL（Ubuntu）で動かす：wsl.exe -d Ubuntu -- bash /mnt/d/PLATEAU/plateau-recorder/scripts/check_plateau_converter.sh
# nusamai は ~/nusamai/nusamai に展開してある前提（Project-PLATEAU/PLATEAU-GIS-Converter v0.1.14、MIT）。
set -u
NUSAMAI=${NUSAMAI:-$HOME/nusamai/nusamai}
REPO=/mnt/d/PLATEAU/plateau-recorder
GML=${1:-$REPO/out/lod3/dataset/udx/bldg/52365768_bldg_6697_op.gml}
OUT=${2:-$REPO/out/lod3_converted}
mkdir -p "$OUT"
echo "変換ツール: $($NUSAMAI --version)"
echo "入力: $GML"
for sink in geojson 3dtiles gltf; do
  rm -rf "$OUT/$sink"
  echo "== $sink"
  "$NUSAMAI" "$GML" --sink "$sink" --output "$OUT/$sink" --schema "$OUT/schema_$sink.json" > "$OUT/log_$sink.txt" 2>&1
  echo "終了コード $?"
  tail -5 "$OUT/log_$sink.txt"
done
ls -R "$OUT" | head -60

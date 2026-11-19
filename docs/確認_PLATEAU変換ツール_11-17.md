# 確認：PLATEAU 公式の変換ツールで開けるか（2026-11-17）

## 使った物

- PLATEAU GIS Converter のコマンド版 nusamai（Project-PLATEAU/PLATEAU-GIS-Converter、v0.1.14、2026-02-05、MIT）。
- Windows 用のコマンド版は配られていないので、Linux 版（`nusamai-v0.1.14-x86_64-unknown-linux-gnu.tar.gz`、8,305,404 バイト、sha256 `dd472b27…bb63f`）を WSL の Ubuntu で動かした。展開先は WSL の `~/nusamai`、ダウンロードの置き場は `D:\PLATEAU\tools\nusamai-v0.1.14\`。
- 手順は `scripts/check_plateau_converter.sh`（WSL で `bash /mnt/d/PLATEAU/plateau-recorder/scripts/check_plateau_converter.sh`）。

## 結果

| 確かめたこと | 結果 |
|---|---|
| 変換が通るか | GeoJSON・3D Tiles とも終了コード 0、エラーなし（1 秒前後）。glTF は「Pipeline canceled」だった。原因は 11/19 に分かった：glTF は出力を平面直角座標系にしないと止まる（`--epsg 6675`、名古屋は VII 系）。指定すると通る（`scripts/check_plateau_converter.sh` に入れた） |
| 形が残るか | 残る。面の数が書き出した CityGML と同じ（本部棟 155、図書館 111、タワー75 334、共通講義棟南 112）。窓の穴を含む LOD3 の形がそのまま入る |
| 場所 | 正しい（3D Tiles の中心が 北緯 35.136°・東経 136.977°） |
| 建物単位の属性 | 残る。用途・階数・高さ・データ品質（LOD3 の取得の区分・形の出どころ）・私たちが足した「LOD3 の等級の内訳」「精度の物差し」 |
| 窓 1 枚・壁 1 面ごとの属性（σ・等級・出どころ・根拠の写真・外装材） | **消える**。変換ツールが壁・窓を建物 1 つにまとめる作り（3D Tiles と GeoJSON の既定）のため。データの誤りではない。変換の設定は「出力 LOD の選択」と「属性の配列をカンマ区切りに」だけで、まとめ方は変えられない |
| 表示 | 3D Tiles を CesiumJS（PLATEAU VIEW の土台）で表示できた（`viewer/tiles.html`）。建物を選ぶと上の属性が出る |

## 意味

- 書き出した LOD3 は、PLATEAU の公式の変換ツールでそのまま変換でき、PLATEAU VIEW と同じ仕組みで表示できる。
- 値ごとの確かさ（作品の主張の中心）は、公式の変換を通すと建物単位の内訳だけになる。窓 1 枚ごとの根拠を見せるのは、CityGML をそのまま読む私たちの表示画面（`viewer/index.html`）の役目。
- 今後の案：建物単位の属性に、等級の内訳に加えて「写真で確かめた窓の割合」などを足せば、公式の変換の後でも主張の一部が残る。

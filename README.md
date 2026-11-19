# PLATEAU Recorder（表示画面）

https://github.com/yuu737/plateau-recorder の表示画面です。中身は `bash scripts/make_pages.sh` で書き出し済みの生成物から写した物です。

- 表示画面：`viewer/index.html?lod=30|31|32`、3D Tiles（PLATEAU 公式の変換ツール nusamai で変換）：`viewer/tiles.html`
- CityGML の建物は「3D 都市モデル（Project PLATEAU）名古屋市」（国土交通省）を引き継いで LOD を足した物です。屋根の質感は国土地理院「全国最新写真（シームレス）」。
- 写真の切り抜き・壁の質感は、名城大学（広報部門）の許可を得て公開している写真の公開版（掲示物・人名・電話番号・顔・ナンバーを塗りつぶし済み）から作りました。
- コードは MIT License、写真と生成物の条件は本体のリポジトリの `LICENSE-DATA.md` に従います。

# rri-baselines

MMECG 公開データ上で, ミリ波レーダから RRI を推定する先行手法を, 提案手法(公開実装 mmwave-rri)と同じ条件で再現・比較する.

## そろえた条件(src/rri_baselines/protocol.py)

| 項目 | 内容 |
|---|---|
| 入力 | 公開 MMECG の RCG 50 点(胸壁の加速度, 200 Hz). 前処理済みの試行は mmwave-rri の `data/processed` を読む |
| 分割 | 11 名を 1 名ずつ test に残す LOSO. fold i は test が i 番目, val が (i+1) 番目の参加者, 学習は残り 9 名 |
| 目標量 | 4 s 窓を 0.2 s ずつ進め, 窓の中央をまたぐ 1 拍の RR. 両端の R 波(NeuroKit2)が窓の中にある窓だけ正解を持つ |
| 採点 | 正解を持つ窓すべての絶対誤差(除外なし) → 試行の平均 → 参加者の平均 → 11 名の平均 |

公開データは加速度 50 点しかないので, 生のレーダ信号から位相を取る前段は再現できない. 比較は「胸壁の運動信号から間隔を出す段」に限る.
採点の実装は, 提案手法の予測を通して公開実装の値 7.019 ms を再現する(`tests/test_protocol.py`).

## 手法と, 論文で決まらない点の選び方

論文で決まらないパラメータは, 相手に有利な側に置くか, 学習側(test 以外)の val で選ぶ. 詳しい仕様は
`radar-hrv-research/literature/docs/baselines/specs/` にある.

| 手法 | 系統 | 実装 | 主な選び方 |
|---|---|---|---|
| EQ-Radio(Zhao ら, MobiCom 2016) | 学習なし. テンプレートと動的計画法で拍を区切る | `eqradio/` | 点は冒頭 30 s で拍の形が最もそろう点. 区間長の範囲・初期値・拍の時刻と R 波のずれは test 以外の 10 名の誤差で選ぶ(ずれの選択は正解を使う) |
| Wu, Sakamoto ら(IEEE Access 2019) | 学習で三角波 2 本(R 波・S 波に頂点)を出し, 極値の間隔を取る | `wu2019/` | 網は論文の図 1 のまま(23,282 パラメータ). 入力は 6.48 s で 4 s 窓の外の実信号も見る. 極値の最小間隔・平滑化・S 波を使うかは val で選ぶ |
| MIBINET(Murshed ら, Array 2023) | CNN が拍の間隔を直接回帰する | `mibinet/` | 網・損失・学習率は論文どおり(論文の形で 1,093,423 パラメータを再現). 50 点を入力チャネルにする. 200 エポック. 論文は推論時に心電図の R 波で窓を切るが, ここでは 4 s 固定窓 |
| radarODE-MTL(Zhang ら, IEEE TIM 2025), 等重み | ECG の形・R 波の位置・心周期長の 3 タスク. 心周期長のヘッドが PPI を出す | `radarode/` | モデル・損失・学習器は radarODE-MTL リポジトリの `repro/` をそのまま使う. SST は MATLAB の wsst を移した `radarode/wsst.py`(試行全体に変換 → 0.72〜100 Hz の 71 行 → 30 Hz → 試行全体の正規化 → 上下反転 → 窓ごとの正規化. 著者のデータ生成コードの順). 20 エポック |

## 結果(2026-10-03 更新, `results/summary.md`)

| 手法 | 参加者平均 MAE [ms] | 原論文の報告値(条件が違う) |
|---|---|---|
| 提案手法(公開実装 91131 相当) | **7.02** | — |
| radarODE-MTL(等重み, 20 エポック) | 17.87 | PPI 誤差 8.0 ms(本文: MMECG 11 分割・ストライド 1 s・200 エポック・5 回平均) |
| EQ-Radio(学習なし) | 45.63 | IBI 誤差 3.2 ms(自前データ) |
| MIBINET | 52.16(中央値 3 で 49.99) | RMSE 20.69 ms(推論時に R 波で窓を切る) |
| Wu・Sakamoto | 88.49 | RMSE 20.6 ms(自前 7 名, 被験者ごとの学習) |
| radarODE-MTL(等重み, 200 エポック・周期 100, 我々ルール) | **18.48** | 同上 |
| TCN(改訂 2 版) | 26.98 | — |
| WaveNet(改訂 2 版) | 27.34 | — |
| Transformer(改訂 2 版) | 78.84 | — |
| 定数(学習側の平均 RR. 内部の確認用で, 論文の比較には入れない) | 100.97 | — |

- 提案手法は 11 名全員で最良.
- **radarODE-MTL の 09-28 の値(24.18 / 26.82 ms)は SST の帯域の取り違えによる誤りで, 外に出さない**(表には「(誤)」をつけて残す).
  原著の MMECG_to_SST.m は wsst の行の後半 = 0.72〜100 Hz を取るが, 再現は ssqueezepy(行が高い方から並ぶ)で同じ「後半」= 0.2〜7.7 Hz を取っていた.
  学習なしの解剖(窓の包絡の自己相関で RR を読む. `results/radarode_band_dissection.md`)では, 40 ms 以内に入る窓が旧帯域 1〜11 %, 新帯域 71〜95 %.
- **論文値の確認**: 公開実装の分割(試行 75〜85 を test, 同じ参加者の他の試行 57〜74・86〜88 は学習側, val なしで test の最小のエポックを採る)で
  20 エポック学習すると 7.61 ms(最終エポック 8.01)で, 論文の 8.0 ms と並ぶ. 論文どおりの 200 エポック(コサイン周期 100)は確認中(`runs/radarode_ew_wsst_official200`).
  同じ試行 75〜85 を LOSO の予測で測ると 10.8 ms(窓平均)で, リークの効果が約 3 ms. 残りの差(LOSO 11 名平均 17.9 ms まで)は参加者の難しさ
  (参加者 13 が 37 ms, 参加者 30 が 25 ms. この 2 名は論文の test に入っていない).
- 公開実装の PPI 指標は損失と 2 標本ずれている(完璧でも 10 ms と出る). 上の値は損失と整合する読み出し(argmax + 1). 公開実装の式では
  LOSO 20.1 ms, 公開分割 12.1 ms. 論文がどちらの式かは本文から決まらない.
- 試行の状態ごと(`results/by_status.md`): radarODE-MTL は運動後(PE, 正解 RR 平均 713 ms)で 35.8 ms(NB 13.4, IB 21.7, SP 11.8). 提案手法は PE で 11.1 ms.
- エポック数: 20 エポック(コサイン周期 20)の val の最小は 11 分割中 9 分割で 16〜18 エポック目. 200 エポック(周期 100)で動くかは参加者 29・30 で確認中
  (`runs/radarode_ew_wsst200`).
- **(10-03)プレーンな網**(`plain/`, `results/plain2/README.md`): 提案手法と同じ窓・正解・val 選択で, 点ごとの共有重み・窓ごとの z スコア・時間平均・勾配クリッピング.
  初版(固定チャネル入力, AI-Huang 版 WaveNet)は一定値並みだったので改訂した(初版の値は `results/plain/`).
- **(10-02)radarODE-MTL 公開分割 200 エポック**: 最小 7.20 ms(epoch 17, val = test), 最終 10.98 ms. 波形・R 波位置は論文の値に合わない
  (波形の PCC 92.6 % は入力を見ない対照 93.6 % と同じ, R 波の時間誤差 35 ms は公開デコーダの性質. `results/radarode_task_metrics.md`).
- **(10-03)彼らのルール 7.20 ms と我々ルール 18.48 ms の差の分解**(`results/radarode_gap_decomp.md`, 学習なし. 既存の予測と記録だけ):
  ① 評価に使う試行・参加者 +6.3 ms(公開分割の test は参加者 29・30 の 11 試行だけで, 公開コードは試行 1〜88 しか読まず参加者 30 の運動後 89〜91 を落とす. この 3 試行だけで +1.3〜4.8),
  ② 被験者リーク +2.5 ms(同じ参加者の他の試行を学習に含む), ③ 集計(参加者を等しく重みづけ)+1.5 ms(順序により 0.1〜2.0). 次点は test でのエポック選択 +1.2 ms.
- **(10-03)radarODE-MTL 我々ルール 200 エポック 確定 18.48 ms**(`results/radarode_loso200_final.md`. 公開式 22.13, 0.8 s 刻み 18.72).
- **(10-04)radarODE-MTL 再現の出所一覧**(`results/radarode_provenance.md`). 全部品・設定を 公開実装 / 論文 / CFT-RFcardi のコード / 配布重み / 我々の推定 に分類. backbone は配布重みと構造が完全一致, 10 点入力で論文の計算量に 1〜2 % で一致, params は −12 %.
  暫定 17.68 との差は参加者 1 だけ(val 最良が epoch 23 → 163 に移り test 14.9 → 23.6 ms). val 最良は 11 分割中 10 分割が epoch 12〜32.
  波形・R 波位置(`results/radarode_ew_wsst200_tasks.md`): RMSE 0.118(論文 0.125)・PCC 89.0 %(79.7)・R² 0.51(0.63)・時間誤差 43.8 ms(9.7)・MDR 33.6 %(5.51).
  波形は入力を見ない対照(RMSE 0.106・PCC 89.1 %・R² 0.62)を上回らない. 波形の val 最良は 6 分割で epoch 4〜34 に来ており,
  タスク別の保存を入れる前(10-02)に過ぎていたため, その 6 分割は保存できた後半のエポックで評価した(表に両方のエポックを併記).
  論文 Table V の 8.12 M パラメータに一致する実装は公開リポジトリの全 58 コミットに無い(5 時点で組み立てても 37.7 M のまま).
- Wu・Sakamoto の復号(極値の最小間隔・平滑化・S 波の使用)は val で選ぶ. 格子を広げる前は 84.54 ms, 論文どおりの固定復号では 444.9 ms.

## 再現できない部分

- 生のレーダ信号から位相を取る前段(公開データは加速度 50 点だけ). EQ-Radio・Wu の報告値はこの段を含む.
- radarODE-MTL の SST は MATLAB の wsst そのものではなく移植(`radarode/wsst.py`). MATLAB の出力との数値の照合はしていない(MATLAB が手元に無い). 時間方向の間引きは resample の代わりに resample_poly.
- MIBINET の推論時の R 波による窓切りと 7 窓平均(心電図を使うので, 心電図なしの目標とは合わない).

## 使い方

```bash
uv sync --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m rri_baselines.eqradio.window compute 8 && .venv/bin/python -m rri_baselines.eqradio.window evaluate
.venv/bin/python -m rri_baselines.wu2019.run train 0 1 2 3 4 5 6 7 8 9 10 && .venv/bin/python -m rri_baselines.wu2019.run evaluate
.venv/bin/python -m rri_baselines.mibinet.run train 0 1 2 3 4 5 6 7 8 9 10 && .venv/bin/python -m rri_baselines.mibinet.run evaluate
../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode.wsst cuda   # SST のキャッシュ(約 7 GB)
RADARODE_ROOT=../radarODE-MTL ../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode.train <分割> 20
.venv/bin/python -m rri_baselines.report      # 比較表 results/summary.md
.venv/bin/python -m rri_baselines.by_status   # 試行の状態ごとの誤差 results/by_status.md
```

## radarODE(Zhang ら, IEEE TMC 2025)の長期再構成まで(`radarode_long/`, 2026-10-01, 学習前)

本線は「長期モジュールが出す 4 s の ECG に NeuroKit2(`ecg_rri.window_rr`)」. 入力は評価する 4 s 窓の中だけ(SST も窓ごとにかける. 拍を中心にした 4 s の SST は窓の外を 0 で埋める).

| 段 | 実装 | 原文 |
|---|---|---|
| SST | `sst.py`. MATLAB wsst の移植(`radarode/wsst.py` と float64 で差 0)を窓 800 点にかけ, [1, 25] Hz の 42 行(23.67〜1.06 Hz)を 71 行へ補間, 30 Hz(120 フレーム), 点ごとに [0, 1] | p.6・p.9 "[1, 25] Hz", Table I の 71 行 |
| PPI(Algorithm 1) | `ppi.py`. 50 点のエネルギー波形 → 区間ごとに NeuroKit2 のピーク検出 → 差分の候補に KDE(h = n^(-1/5)) | p.5〜6, 式 17〜18 |
| SCEG | radarODE-MTL の `repro/` の Backbone + ShapeDecoder(ODE あり). 段幅は Table I の 128/256/512/1024 | Fig. 5, Table I・II |
| 形態参照 | `data.py`. SCEG の断片を KDE の PPI に伸縮し, エネルギー波形の位相で置いて連結 | p.8 |
| 長期 | `model.py` の LongTerm. Table III の encoder・decoder + 膨張畳み込み 9 段 | Table III, Fig. 7 |

- KDE の PPI を窓中央の RR とみなした参考値(学習なし, 全 11 名): **51.45 ms**(設定を test 以外の 10 名で選ぶと 11 分割すべて [1, 25] Hz・l_seg 4 s・NeuroKit2 の ECG 検出器・Scott の幅).
  val の 1 名で選ぶと 70.98 ms(参加者 17 で 187 ms に崩れる). 候補の平均(論文 Fig. 9 の Mean)は 204 ms. `results/radarode_long/ppi_kde.json`
- 計算量: `results/radarode_long/complexity.json`(FLOPpy). SCEG 29.3 M パラメータ・12.1 GFLOPs/拍, 長期 3.17 M・1.07 GFLOPs/窓, 窓 1 つ(平均 5.88 拍)で 72.3 GFLOPs.
- 近道の確認(`shortcut_check.py`): 正解に近い参照で学習すると長期モジュールはレーダを使わず参照を写す(レーダを入れ替えても RMSE 0.0336 → 0.0340).
- 使い方(s7 の CPU で KDE, 学習は GPU):

```bash
python -m rri_baselines.radarode_long.ppi compute --device cpu --workers 16 && python -m rri_baselines.radarode_long.ppi evaluate
RADARODE_ROOT=../radarODE-MTL ../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode_long.train all <分割>    # sceg → refs → long → eval
RADARODE_ROOT=../radarODE-MTL ../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode_long.flops
```

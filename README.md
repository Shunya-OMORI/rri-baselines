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
| radarODE-MTL(Zhang ら, IEEE TIM 2025), 等重み | ECG の形・R 波の位置・心周期長の 3 タスク. 心周期長のヘッドが PPI を出す | `radarode/` | モデル・損失・学習器は radarODE-MTL リポジトリの `repro/` をそのまま使う. SST は原著の MMECG_to_SST.m の順序で, wsst を ssqueezepy で代用. 20 エポック |

## 結果(2026-09-28, `results/summary.md`)

| 手法 | 参加者平均 MAE [ms] | 原論文の報告値(条件が違う) |
|---|---|---|
| 提案手法(公開実装 91131 相当) | **7.02** | — |
| radarODE-MTL(等重み) | 24.18 | PPI 誤差 8.0 ms(MMECG・4 s 窓・ストライド 1 s・200 エポック・5 回平均) |
| EQ-Radio(学習なし) | 45.63 | IBI 誤差 3.2 ms(自前データ) |
| MIBINET | 52.16(中央値 3 で 49.99) | RMSE 20.69 ms(推論時に R 波で窓を切る) |
| Wu・Sakamoto | 88.49 | RMSE 20.6 ms(自前 7 名, 被験者ごとの学習) |
| 定数(学習側の平均 RR) | 100.97 | — |

- 提案手法は 11 名全員で最良(参加者 2 だけは radarODE-MTL と 6.3 対 6.4 でほぼ同じ).
- 試行の状態ごと(`results/by_status.md`): radarODE-MTL は運動後(PE, 正解 RR 平均 713 ms)で 53.1 ms と崩れ, 窓の 12.7 % が
  正解の 1.2〜1.6 倍(倍周期 1.6〜2.4 倍は 2.2 %). 短い RR を学習側の分布へ寄せて長めに出す. 提案手法は PE で 11.1 ms.
- radarODE-MTL は論文の 200 エポックを 20 に縮めた(計算時間). val で選ばれたエポックは 3〜19 に散らばり, 最終エポックは 11 分割中 1 つだけ.
- Wu・Sakamoto の復号(極値の最小間隔・平滑化・S 波の使用)は val で選ぶ. 格子を広げる前は 84.54 ms, 論文どおりの固定復号では 444.9 ms.

## 再現できない部分

- 生のレーダ信号から位相を取る前段(公開データは加速度 50 点だけ). EQ-Radio・Wu の報告値はこの段を含む.
- radarODE-MTL の SST は MATLAB の wsst の代わりに ssqueezepy. 行の並びは分からず, 0.2〜7.7 Hz・窓ごとの正規化(val の誤差が小さい側)に置いた.
- MIBINET の推論時の R 波による窓切りと 7 窓平均(心電図を使うので, 心電図なしの目標とは合わない).

## 使い方

```bash
uv sync --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m rri_baselines.eqradio.window compute 8 && .venv/bin/python -m rri_baselines.eqradio.window evaluate
.venv/bin/python -m rri_baselines.wu2019.run train 0 1 2 3 4 5 6 7 8 9 10 && .venv/bin/python -m rri_baselines.wu2019.run evaluate
.venv/bin/python -m rri_baselines.mibinet.run train 0 1 2 3 4 5 6 7 8 9 10 && .venv/bin/python -m rri_baselines.mibinet.run evaluate
.venv/bin/python -m rri_baselines.radarode.sst 4
RADARODE_ROOT=../radarODE-MTL ../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode.train <分割> 20
.venv/bin/python -m rri_baselines.report      # 比較表 results/summary.md
.venv/bin/python -m rri_baselines.by_status   # 試行の状態ごとの誤差 results/by_status.md
```

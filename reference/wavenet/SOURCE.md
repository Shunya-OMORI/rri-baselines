AI-Huang/WaveNet(Kan Huang, Keras/PyTorch による WaveNet の実装. 原論文 van den Oord ら 2016, arXiv:1609.03499)の
wavenet/torch_fn/wavenet.py をそのまま写したもの. コミット 32ab6cb816624ead4e42e07e963b2971145a6a61(2022-01-24), MIT ライセンス(LICENSE). 取得 2026-10-01(ユーザ指定).
使うのは WaveNetLayer・WaveNetBlock(膨張畳み込み + tanh×sigmoid のゲート + 1 チャネルのスキップ + 残差). 上段(スキップの和 → ReLU → 1×1 → ReLU → 1×1 → 平坦化 → 全結合)は
同ファイルの WaveNet.forward と Keras 版(wavenet/keras_fn/wavenet.py の WaveNet)の順のまま, 出力を softmax 256 から RR 1 個に替える.
同ファイルの WaveNet クラス自体は super().__init__() が無く torch.sum(list) で落ちるので使わない. リポジトリは filters・kernel_size・n の値を示していない.

% MATLAB で 1 回だけ走らせる照合用の保存(試行 1 の点 1). 著者の MMECG_to_SST.m と同じ呼び方.
% 出力 wsst_check_trial1_p1.mat (約 160 MB, -v7) を compare_with_matlab_mat.py に渡す.
load('1.mat');                       % MMECG の公開データ finalPartialPublicData20221108/1.mat
x = data.RCG(:,1);
ver = version;                       % 版で wsst の行数とアルゴリズムが変わる(R2024a で cwtfilterbank ベースに)
[sst,f] = wsst(x,200,'VoicesPerOctave',10);
freq_len = length(sst(:,1));
half = abs(sst(freq_len/2:freq_len,:));
R = zeros(size(half,1), ceil(size(half,2)*30/200));
for j = 1:size(half,1)
    R(j,:) = resample(half(j,:).',30,200);
end
[~,b] = resample(half(1,:).',30,200);   % 既定のフィルタ係数
S = (R - min(R(:))) / (max(R(:)) - min(R(:)));
save('wsst_check_trial1_p1.mat','x','sst','f','half','R','S','b','ver','-v7');

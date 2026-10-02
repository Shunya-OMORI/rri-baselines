#!/bin/bash
# s7 で動かす見張り(2026-10-02). mm-JEPA-ECG(公式実装)の 11 分割を, 自分(omori26)の GPU が合計 2 枚未満になったときだけ
# 1 分割ずつ投入し, 各分割の DiT が 20 エポックに達したら同じ GPU で途中の結果(select/test --upto 20)を出す.
#
# - 上限は「自分の全ジョブ(radarODE-MTL などを含む)が使う GPU の合計 2 枚」(ユーザ方針 2026-10-02: 他の人のために 2 枚だけ使う).
#   他の人のプロセスが載っている GPU, 使用中の GPU は空きと数えない. 投入直後でまだ GPU を掴んでいない分割も枠として数える.
# - 失敗した分割は再投入しない(failed_<i> を残して次へ). すべて終わるか失敗したら抜ける.
# - 外部の状態(共有サーバの GPU の空き)を待つため 5 分ごとに見る. 終了の待ち合わせではない.
#
#   setsid nohup bash ~/rri-baselines/scripts/s7_mmjepa_watch.sh > ~/jobs/mmjepa_watch/watch.log 2>&1 < /dev/null &
set -u
MAX_MINE=2
FOLDS="0 1 2 3 4 5 6 7 8 9 10"
ST=$HOME/jobs/mmjepa_watch
REPO=$HOME/rri-baselines
RUNS=$REPO/runs/mmjepa_official
mkdir -p "$ST"
ENV='export RRI_DATA_ROOT=$HOME/mmwave-rri-91131/data/processed MMJEPA_OFFICIAL=$HOME/external/mmJEPA-ECG PYTHONPATH=$HOME/pylib/mmjepa_official:src PYTHONDONTWRITEBYTECODE=1 MMJEPA_MAX_WORKERS=1 MMJEPA_DEVICE=cuda:0'

log() { echo "[$(date '+%F %T')] $*"; }

# 自分の GPU の番号(プロセスが載っているもの)と, 走行中の自分のジョブに割り当てた番号の和集合
my_gpus() {
    declare -A idx_of
    while IFS=', ' read -r i u; do idx_of[$u]=$i; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    {
        nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read -r p u; do
            [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && echo "${idx_of[$u]}"
        done
        for m in "$ST"/run_*/gpu; do
            [ -f "$m" ] || continue
            d=$(dirname "$m"); [ -f "$d/rc" ] && continue
            kill -0 "$(cat "$d/pid" 2>/dev/null)" 2>/dev/null && cat "$m"
        done
    } | sort -u | grep -c .
}

# 誰のプロセスも載っておらず, 使用量・使用率が低く, 空きが 20 GB 以上の GPU を 1 つ(空きが最大のもの)
free_gpu() {
    local used_uuids; used_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    local reserved=" $(for m in "$ST"/run_*/gpu; do [ -f "$m" ] && [ ! -f "$(dirname "$m")/rc" ] && cat "$m"; done | tr '\n' ' ') "
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
    while IFS=', ' read -r idx uuid used total util; do
        echo "$used_uuids" | grep -qx "$uuid" && continue
        case "$reserved" in *" $idx "*) continue ;; esac
        [ "$used" -ge 1000 ] && continue; [ "$util" -ge 10 ] && continue
        [ $((total - used)) -ge 20000 ] || continue
        echo "$((total - used)) $idx"
    done | sort -rn | head -1 | awk '{print $2}'
}

launch() {   # $1 名前, $2 GPU, $3 コマンド
    local d="$ST/run_$1"
    mkdir -p "$d"; rm -f "$d/rc" "$d/pid"
    echo "$2" > "$d/gpu"
    printf 'cd %q || exit 97\n%s\nexport CUDA_VISIBLE_DEVICES=%s\n%s\n' "$REPO" "$ENV" "$2" "$3" > "$d/job.sh"
    (cd "$d" && setsid nohup bash -c 'bash job.sh > log 2>&1; echo $? > rc' < /dev/null > /dev/null 2>&1 & echo $! > "$d/pid")
    log "起動 $1 (GPU $2)"
}

fold_dir() { ls -d "$RUNS"/fold$(printf %02d "$1")_test* 2>/dev/null | head -1; }

below=0
log "見張り開始(自分の GPU の上限 $MAX_MINE 枚, 分割 $FOLDS)"
while :; do
    remaining=0
    for i in $FOLDS; do
        r="$ST/run_f$i"
        if [ -f "$r/rc" ] && [ ! -f "$ST/done_$i" ] && [ ! -f "$ST/failed_$i" ]; then
            if [ "$(cat "$r/rc")" = 0 ]; then touch "$ST/done_$i"; log "完了 分割 $i"; else touch "$ST/failed_$i"; log "失敗 分割 $i (rc=$(cat "$r/rc"), ログ $r/log)"; fi
        fi
        # 途中の結果: DiT の epoch_020.pt ができたら, その分割の GPU で select/test --upto 20
        fd=$(fold_dir "$i")
        if [ -n "$fd" ] && [ ! -e "$ST/run_mid$i" ] && ls "$fd"/e2e/*-e2e/checkpoints/epoch_020.pt >/dev/null 2>&1; then
            launch "mid$i" "$(cat "$r/gpu")" ".venv/bin/python -m rri_baselines.mmjepa_official.run select $i --upto 20 --device cuda:0 && .venv/bin/python -m rri_baselines.mmjepa_official.run test $i --upto 20 --device cuda:0"
        fi
        [ -f "$ST/done_$i" ] || [ -f "$ST/failed_$i" ] || remaining=$((remaining + 1))
    done
    [ "$remaining" -eq 0 ] && { log "全分割が終わった(失敗: $(ls "$ST" | grep -c '^failed_'))"; break; }
    # 投入: 自分の GPU が上限未満で, 空いている GPU があり, まだ投入していない分割があるとき 1 つだけ
    next=""; for i in $FOLDS; do [ -e "$ST/run_f$i" ] || { next=$i; break; }; done
    if [ -n "$next" ]; then
        mine=$(my_gpus)
        # 自分の別ジョブ(radarODE-MTL)は再開のために一瞬 GPU を離れる(10-02 19:10, 1 分弱). 2 回続けて(5 分以上)上限未満のときだけ投入する
        if [ "$mine" -lt "$MAX_MINE" ]; then below=$((below + 1)); else below=0; fi
        if [ "$below" -ge 2 ]; then
            g=$(free_gpu)
            if [ -n "$g" ]; then
                launch "f$next" "$g" ".venv/bin/python -m rri_baselines.mmjepa_official.run all $next --device cuda:0 && .venv/bin/python -m rri_baselines.mmjepa_official.run cleanup $next"
                below=0
            fi
        fi
    fi
    sleep 300
done

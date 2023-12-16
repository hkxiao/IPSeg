#! /bin/bash

start_program() {
    export CUDA_VISIBLE_DEVICES=$1
    python  opensam_fss.py --sam_type vit_h  \
        --data /data/tanglv/data/fss-te/$2 \
        --ref_txt ref_composed2.txt \
        --erosion \
        --sd_weight=0.1 \
        --sd_layer_weight=0.3,0.2,0.1
}

# 设置GPU列表
CUDA_VISIBLE_DEVICES_LIST=(1)
datasets=(fss)
PID_LIST=()
STATUS=()

# run
for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
do
    echo "Start: ${start[i]}"
    echo "End ${end[i]}"
    echo "GPU ${CUDA_VISIBLE_DEVICES_LIST[i]}"
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} ${datasets[i]}  &
    PID_LIST+=($!)
    STATUS+=(-1)
done

#check crash
while true
do
    finish=true
    for i in $(seq 0 $((${#CUDA_VISIBLE_DEVICES_LIST[@]}-1)))
    do
        process_id=${PID_LIST[i]}
        # 检查进程是否在执行
        if [ ${STATUS[i]} -eq 0 ]; then
            echo "进程 $process_id 执行完成"
            continue
        elif ps -p $process_id > /dev/null; then
            echo "进程 $process_id 正在执行"
            finish=false
        fi
    done
    if [ $finish == true ]; then
        break
    fi
    sleep 5
done
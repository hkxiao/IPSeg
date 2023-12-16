#! /bin/bash

start_program() {
    python  utils/make_refimg_openvoc_patch.py --seed $1 --dataset VOC2012
}

# 设置GPU列表
seed_list=(8 9 10 11 12 13 14 15)

# run
for i in $(seq 0 $((${#seed_list[@]}-1)))
do
    start_program ${CUDA_VISIBLE_DEVICES_LIST[i]} ${seed_list[i]}  &
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
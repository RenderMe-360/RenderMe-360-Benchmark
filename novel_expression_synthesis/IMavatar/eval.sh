source /mnt/cache/pandongwei/scripts/cuda11.0_cudnn8.0.5_env.sh

cd code
SEQS=(audio_audio_multiflame)

for((i=0;i<${#SEQS[@]};++i)); do
    TAG=IMavatar-${SEQS[i]}
    conf_dir='../data/conf/'${SEQS[i]}
    for file in $(ls ${conf_dir}); do
        echo $file
        python scripts/exp_runner.py \
            --conf ../data/conf/${SEQS[i]}/${file} \
            --is_eval \
            --checkpoint latest \
        2>&1 | tee ../data/experiments/logs/${SEQS[i]}/${file}_eval.txt
    done
done
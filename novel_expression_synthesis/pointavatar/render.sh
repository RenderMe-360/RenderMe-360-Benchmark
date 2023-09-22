
cd code
SEQS=(audio_audio_multiflame)

for((i=0;i<${#SEQS[@]};++i)); do
    TAG=IMavatar-${SEQS[i]}
    conf_dir='../data/conf/'${SEQS[i]}
    mkdir ../data/experiments/logs/${SEQS[i]}_eval
    for file in $(ls ${conf_dir}); do
        echo $file
        python scripts/exp_runner.py \
            --conf ../data/conf/${SEQS[i]}/${file} \
            --is_eval \
        2>&1 | tee ../data/experiments/logs/${SEQS[i]}_eval/${file}.txt
    done
done
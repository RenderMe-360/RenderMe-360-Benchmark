
mkdir data/experiments
mkdir data/experiments/logs

cd code

SEQS=(audio_audio_multiflame)
for((i=0;i<${#SEQS[@]};++i)); do
    mkdir ../data/experiments/logs/${SEQS[i]}
    TAG=IMavatar-${SEQS[i]}
    conf_dir='../data/conf/'${SEQS[i]}
    for file in $(ls ${conf_dir}); do
        echo $file
        python scripts/exp_runner.py --conf ../data/conf/${SEQS[i]}/${file} --nepoch 100 \
        2>&1 | tee ../data/experiments/logs/${SEQS[i]}/${file}.txt
    done
done
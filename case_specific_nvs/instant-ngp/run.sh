STEPS=30000
SINGLE=0
MULTI=1
TRAIN=1
RENDER=0

LOG=data_NVS_instant-ngp
# test_views=(01 13 25 40) # setting 1: 56 train views, 4 test views
test_views=(02 06 08 10 12 14 18 22 26 30 34 36 38 42 44 46 50 51 52 54 56 58) # setting 2: 38 train views, 22 test views
# test_views=(01 03 04 05 07 09 11 13 15 16 17 19 21 23 25 27 28 29 31 33 35 37 39 40 41 43 45 47 49 51 53 55 57 59) # setting 3: 26 train views, 34 test views

if [[ $SINGLE -eq 1 ]]; then
# single runing
SET=set1
SEQ=0100
subject=0100_e_10
echo ./${LOG}/${SET}/${SEQ}/${subject}
if [[ $TRAIN -eq 1 ]]; then
    python scripts/run.py \
      --datadir ./${LOG}/${SET}/${SEQ} \
      --subject ${subject} \
      --n_steps ${STEPS} \
      --test_views ${test_views[*]} \
      --save_ckpt \
      --render \
      --save_mesh \
    2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_train.txt
fi
if [[ $RENDER -eq 1 ]]; then
    python scripts/run.py \
      --datadir ./${LOG}/${SET}/${SEQ} \
      --subject ${subject} \
      --n_steps 0 \
      --test_views ${test_views[*]} \
      --resume \
      --save_mesh \
    2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_render.txt
fi
fi

if [[ $MULTI -eq 1 ]]; then
# multi scripts/runing
SET=set1
SEQS=(0100  0026 0099 0094 0278  0295 0290 0259  0297  0189)

for ((i=0; i<${#SEQS[@]}; ++i)) do
  SEQ=${SEQS[i]}
  # for subject in `ls  ./${LOG}/${SET}/${SEQ}/`
  for j in $(seq 10 11)
  do
    subject=${SEQ}_e_${j}
    echo ./${LOG}/${SET}/${SEQ}/${subject}
    if [[ $TRAIN -eq 1 ]]; then
        python scripts/run.py \
          --datadir ./${LOG}/${SET}/${SEQ} \
          --subject ${subject} \
          --n_steps ${STEPS} \
          --test_views ${test_views[*]} \
          --save_ckpt \
          --render \
          --save_mesh \
        2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_train.txt &
    fi
    if [[ $RENDER -eq 1 ]]; then
        python scripts/run.py \
          --datadir ./${LOG}/${SET}/${SEQ} \
          --subject ${subject} \
          --n_steps 0 \
          --test_views ${test_views[*]} \
          --resume \
          --save_mesh \
        2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_render.txt &
    fi
    sleep 0.3
  done
done

SET=set2
SEQS=(0041 0168 0175 0253 0250)

for ((i=0; i<${#SEQS[@]}; ++i)) do
  SEQ=${SEQS[i]}
  for j in $(seq 10 11)
  do
    subject=${SEQ}_e_${j}
    echo ./${LOG}/${SET}/${SEQ}/${subject}
    if [[ $TRAIN -eq 1 ]]; then
        python scripts/run.py \
          --datadir ./${LOG}/${SET}/${SEQ} \
          --subject ${subject} \
          --n_steps ${STEPS} \
          --test_views ${test_views[*]} \
          --save_ckpt \
          --render \
          --save_mesh \
        2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_train.txt &
    fi
    if [[ $RENDER -eq 1 ]]; then
        python scripts/run.py \
          --datadir ./${LOG}/${SET}/${SEQ} \
          --subject ${subject} \
          --n_steps 0 \
          --test_views ${test_views[*]} \
          --resume \
          --save_mesh \
        2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_render.txt &
    fi
    sleep 0.3
  done
done

SET=set3
SEQS=(0116 0156 0048 0262 0232 0195)

for ((i=0; i<${#SEQS[@]}; ++i)) do
  SEQ=${SEQS[i]}
  for j in $(seq 10 11)
  do
    subject=${SEQ}_e_${j}
    echo ./${LOG}/${SET}/${SEQ}/${subject}
    if [[ $TRAIN -eq 1 ]]; then
        python scripts/run.py \
          --datadir ./${LOG}/${SET}/${SEQ} \
          --subject ${subject} \
          --n_steps ${STEPS} \
          --test_views ${test_views[*]} \
          --save_ckpt \
          --render \
          --save_mesh \
        2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_train.txt &
    fi
    if [[ $RENDER -eq 1 ]]; then
        python scripts/run.py \
          --datadir ./${LOG}/${SET}/${SEQ} \
          --subject ${subject} \
          --n_steps 0 \
          --test_views ${test_views[*]} \
          --resume \
          --save_mesh \
        2>&1 | tee ./${LOG}/${SET}/${SEQ}/${subject}/log_render.txt &
    fi
    sleep 0.3
  done
done
fi

IMGDIR='./data_NVS_mvp_nv'

LOG=logs_exp
SEQS=(0026 0094 0297 0290 0253 0175 0156 0262 0048 0100 0259 0099 0278 0295 0189 0041 0168 0250 0116 0232 0195)
MODE=exp


for((i=0;i<${#SEQS[@]};++i)); do
    TAG=nv-${SEQS[i]}-train-${LOG}
    python train.py config_renderme360.py \
    --datadir ${IMGDIR} \
    --annotdir ${IMGDIR}/${SEQS[i]}/annots.npy \
    --outdir ./${LOG} \
    --subject ${SEQS[i]} \
    --mode ${MODE} \
    --ws_factor 0.6 \
    2>&1 | tee ./${LOG}/log_train_${SEQS[i]}.txt &
	sleep 0.5	
done

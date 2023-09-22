IMGDIR='./data_NVS_mvp_nv'

LOG=logs_exp
SEQS=(0026 0094 0297 0290 0253 0175 0156 0262 0048
	0100 0259 0099 0278 0295 0189 0041 0168 0250 0116 0232 0195)
MODE=exp
cams=(01 03 04 05 07 09 11 13 15 16 17 19 21 23 25 27 28 29 31 33 35 37 39 40 41 43 45 47 49 51 53 55 57 59)

FIX_VIEW=0 #  rendering the result of surrounding views
FIX_VIEW=1 # rendering the result of the fix view

for((i=0;i<${#SEQS[@]};++i)); do
	if [[ $FIX_VIEW -eq 1 ]]; then
		for((j=0;j<${#cams[@]};++j)); do
			TAG=eval-nv-${SEQS[i]}-${cams[j]}
			echo ${TAG}
      python render.py config_renderme360.py \
      --datadir ${IMGDIR} \
      --annotdir ${IMGDIR}/${SEQS[i]}/annots.npy \
      --outdir ./${LOG} \
      --subject ${SEQS[i]} \
      --mode ${MODE} \
      --cam ${cams[j]} \
      --move_cam 0 \
      --ws_factor 0.6 \
      2>&1 | tee ./${LOG}/log_eval_${DATES[i]}_${SEQS[i]}.txt
		done
	else
		TAG=eval-nv-${SEQS[i]}
		echo ${TAG}
    python render.py config_renderme360.py \
    --datadir ${IMGDIR} \
    --annotdir PARAMS/${DATES[i]}/annots.npy \
    --outdir ./${LOG} \
    --subject ${SEQS[i]} \
    --mode ${MODE} \
    --move_cam 0 \
    --ws_factor 0.6 \
    2>&1 | tee ./${LOG}/log_render_${DATES[i]}_${SEQS[i]}.txt
	fi
done
cd nsff_exp

sets=('singleview')
subjects=(0295_h1_2bk       0189_h1_3bk  0290_h1_2b   0259_h1_7bn            0278_h1_6bk
          0295_h1_3bk          0290_h1_3bn  0094_h1_3bk        0297_h1_2bk
          0295_h1_7b              0259_h1_2b      0094_h1_4bn        0297_h1_3bk
                 0259_h1_3bk     0278_h1_4bn  0297_h1_7bk
        )

for set in ${sets[@]}
do  
    for subject in ${subjects[@]}
    do
        echo ${subject}
        python run_nerf.py \
          --config ../data_hairrendering_nsff_nrnerf/config/nsff/${subject}.txt \
          --final_height 512 \
          --render_lockcam_slowmo --target_idx 15 \
        2>&1 | tee ../experiments/${set}/${subject}/log_eval_render_lockcam_slowmo.txt
    done
done
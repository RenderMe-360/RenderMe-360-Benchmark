sets=('singleview')
subjects=('0295_h1_2bk' '0295_h1_3bk' '0295_h1_7b' '0099_h1_4b' '0099_h1_6bn'  '0290_h1_2b' '0290_h1_3bn' \
        '0094_h1_3bk' '0094_h1_4bn' '0278_h1_4bn' '0278_h1_6bk' \
        '0297_h1_2bk' '0297_h1_3bk' '0297_h1_7bk' '0189_h1_1bk' '0189_h1_3bk' '0189_h1_7b' \
        '0259_h1_2b' '0259_h1_3bk' '0259_h1_7bn')

for set in ${sets[@]}
do  
    mkdir experiments
    mkdir experiments/${set}
    for subject in ${subjects[@]}
    do  
        echo ${subject}
        mkdir experiments/${set}/${subject}
        python train.py --config ./data_hairrendering_nsff_nrnerf/config/nrnerf/${subject}.txt \
            2>&1 | tee ./experiments/${set}/${subject}/logs_train.txt
    done
done
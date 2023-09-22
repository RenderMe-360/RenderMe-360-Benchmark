
sets=('singleview')
subjects=('0295_h1_2bk' '0295_h1_3bk' '0295_h1_7b' '0099_h1_4b' '0099_h1_6bn'  '0290_h1_2b' '0290_h1_3bn' \
        '0094_h1_3bk' '0094_h1_4bn' '0278_h1_4bn' '0278_h1_6bk' \
        '0297_h1_2bk' '0297_h1_3bk' '0297_h1_7bk' '0189_h1_1bk' '0189_h1_3bk' '0189_h1_7b' \
        '0259_h1_2b' '0259_h1_3bk' '0259_h1_7bn')

for set in ${sets[@]}
do
    for subject in ${subjects[@]}
    do
        python free_viewpoint_rendering.py \
            --input experiments/${set}/${subject}/ \
            --output_video_fps 15 \
            --deformations train \
            --camera_path fixed_view_time_interpolation \
            --fixed_view 15
    done
done
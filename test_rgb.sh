python test_models.py something-v1 CHECKPOINT_FILE \
                      --arch bninception --crop_fusion_type avg --test_segments 8 --input_size 0 --test_crops 1 --num_clips 1 \
                      --with_amp -j 8 --save_scores --gsf --gsf_ch_ratio 100
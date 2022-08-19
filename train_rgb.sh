python main.py --dataset something-v1 --split val --arch bninception --num_segments 8 --consensus_type avg \
                           --batch-size 32 --iter_size 1 --dropout 0.5 --lr 0.01 --warmup 10 --epochs 60 \
                           --eval-freq 5 --gd 20 -j 16 \
                           --with_amp --gsf --gsf_ch_ratio 100




# Gate-Shift-Fuse for Video Action Recognition


We release the code and trained models of our paper [Gate-Shift-Fuse for Video Action Recognition](https://arxiv.org/pdf/2203.08897.pdf). If you find our work useful for your research, please cite
```
@article{gsf,
  title={{Gate-Shift-Fuse for Video Action Recognition}},
  author={Sudhakaran, Swathikiran and Escalera, Sergio and Lanz, Oswald},
  journal={arXiv preprint arXiv:2203.08897},
  year={2022}
}
```

### Prerequisites
- Python 3.5
- PyTorch 1.7+

### Data preparation

Please follow the instructions in [GSM repo](https://github.com/swathikirans/GSM) for data preparation.

### Training
```
python main.py --dataset something-v1 --split val --arch bninception --num_segments 8 --consensus_type avg \
               --batch-size 32 --iter_size 1 --dropout 0.5 --lr 0.01 --warmup 10 --epochs 60 \
               --eval-freq 5 --gd 20 -j 16 \
               --with_amp --gsf --gsf_ch_ratio 100
```

### Testing
```
python test_models.py something-v1 CHECKPOINT_FILE \
                      --arch bninception --crop_fusion_type avg --test_segments 8 \ 
                      --input_size 0 --test_crops 1 --num_clips 1 \
                      --with_amp -j 8 --save_scores --gsf --gsf_ch_ratio 100
```
To evaluate using 2 clips and 3 crops, change ``--test_crops 1`` to ``--test_crops 3`` and ``--num_clips 1`` to ``--num_clips 2``. 


### Models

<table style="width:100%" align="center">  
<col width="150">
<tr> 
	<th>Backbone</th>  
	<th>No. of frames</th>  
	<th>SS-v1 Top-1 Accuracy (%)</th>
   
</tr>  
<tr> 
	<td align="center">BNInception</td>
	<td align="center">16</td>  
	<td align="center"><a href='https://drive.google.com/file/d/1O7uEEYXr333YAnh8T5FPg_w2ejEbDqtK/view?usp=sharing'>50.63</a></td>  
</tr>  

<tr> 
	<td align="center">InceptionV3</td>
	<td align="center">16</td>  
	<td align="center"><a href='https://drive.google.com/file/d/1jIymBjJChK5Auj7DRKQuMxGAznJ3pYh9/view?usp=sharing'>53.13</a></td>  
</tr>

<tr> 
	<td align="center">ResNet50</td>
	<td align="center">16</td>  
	<td align="center"><a href='https://drive.google.com/file/d/1nSjhYvsu56e7a9d9VasGo7bxOrJ5j05f/view?usp=sharing'>51.54</a></td>  
</tr>

</table>

All pretrained weights can be downloaded from [here](https://drive.google.com/drive/folders/1KTk5PN0I4wwaOYh4Vi9wTZ-Wq2lSZ7Fb?usp=sharing).


### Acknowledgements

This implementation is built upon the [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch) codebase which is based on [TSN-pytorch](https://github.com/yjxiong/tsn-pytorch). We thank Yuanjun Xiong and Bolei Zhou for releasing TSN-pytorch and TRN-pytorch repos.
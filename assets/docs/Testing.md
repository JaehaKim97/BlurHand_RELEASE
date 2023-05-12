## Pre-trained model

Download [**pre-trained model**](https://drive.google.com/drive/folders/1tf9O-jsoSpH0uYg_XoS1aYia4toaO_Sr?usp=share_link), which is the BlurHandNet trained on BlurHand.

Then, locate `pretrained_BlurHandNet_BH` in `experiments/`.

The testing command is below:

```
CUDA_VISIBLE_DEVICES=0 python src/test.py -opt options/test/pretrained_BlurHandNet_BH.yml
```

In default, it calculates MPJPE on three sequential hands, and MPVPE on current (middle) hand.


|               | MPJPE (past / current / future) | MPVPE (current) |
| ------------- |:-------------:| :-----:|
| BlurHandNet      | 18.04 / 16.78 / 18.18 mm | 15.31 mm|

Note that the value is slightly different from the original paper (18.08 / 16.80 / 18.21, 15.30 in paper), due to reproduction.

The test logs will be saved in ```experiments/exp_name/results```.

(Optional) If you want to visualize mesh sequences into video, set the `visualize_video` option as True. (But it takes a long time.)

## Your own trained model

If you trained your own BlurHandNet following [**training instruction**](Training.md), please run the below command to test your model.

```
CUDA_VISIBLE_DEVICES=0 python src/test.py -opt options/test/BlurHandNet_BH.yml
```

<div align="right">
 <a href="../../README.md" style="float: right;">Link</a> to return main document.
</div>

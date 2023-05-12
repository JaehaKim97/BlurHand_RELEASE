## Training command

For training our BlurHandNet on BlurHand, please run the below command.

```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py -opt options/train/BlurHandNet_BH.yml
```

You can check detailed training configurations in ```options/train/BlurHandNet_BH.yml```.

In default, we used 2 GPUs, but you can change the configurations by modulating `num_gpus` in `.yml` file.

The training states and logs will be saved in ```experiments/BlurHandNet_BH```.

<div align="right">
 <a href="../../README.md" style="float: right;">Link</a> to return main document.
</div>

## Installation

1. Make clone this repo.

```
git clone https://github.com/JaehaKim97/BlurHand_RELEASE.git
cd BlurHand_RELEASE
```

2. Set the environment. We recommend using [**Anaconda**](https://www.anaconda.com/products/distribution).

```
 conda create -n BH_RELEASE python=3.8
 conda activate BH_RELEASE
 pip install -r requirements.txt
```

3. Modify the `torchgeometry` following [**this**](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527). Without modification, you will meet `RuntimeError: Subtraction, the - operator...`.

4. Download `human_model_files.tar.gz` from [**here**](https://drive.google.com/drive/folders/1tf9O-jsoSpH0uYg_XoS1aYia4toaO_Sr?usp=share_link), and unzip the file, then locate `human_model_files` in `/src/utils/human_models`.

5. (Optional) If you are linux user and have external data storage, consider to replace ```experiments```, ```datasets``` as symlink.

## Dataset preparation

Download our [**BlurHand dataset**](BlurHand.md), then locate them on ```datasets```.


<div align="right">
 <a href="../README.md" style="float: right;">Link</a> to return main document.
</div>

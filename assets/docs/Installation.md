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
 python src/setup.py
```

3. Modify the `torchgeometry` following [**this**](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527). Without modification, you will meet `RuntimeError: Subtraction, the - operator...`.

4. Download the MANO model files (```mano_v1_2.zip```) from [**MANO website**](https://mano.is.tue.mpg.de/). Unzip and put ```mano_v1_2/models``` into ```src/utils/MANO```. To be specific, ```MANO_RIGHT.pkl``` and ```MANO_LEFT.pkl``` should be located in ```src/utils/MANO/models/MANO_RIGHT.pkl``` and ```src/utils/MANO/models/MANO_LEFT.pkl```, respectively.



5. (Optional) If you are linux user and have external data storage, consider to replace ```experiments```, ```datasets``` as symlink.

## Dataset preparation

Download our [**BlurHand dataset**](BlurHand.md), then locate them on ```datasets```.


<div align="right">
 <a href="../../README.md" style="float: right;">Link</a> to return main document.
</div>

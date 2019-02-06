# eqip
**E**lectron-microscopy **Q**uasi **I**sotropic **P**rediction scripts

Dependencies are specified as `install_requires` in `setup.py`. On top of that, you need the [CNNectome `quasi-isotropic` branch](https://github.com/hanslovsky/CNNectome/tree/quasi-isotropic) on your `PYTHONPATH`. It is likely, yet untested, that the master branch works as well.


## Install in conda enviornment on GPU cluster:
```sh
conda create -n eqip -c conda-forge -c cpape python=3.6 tensorflow-gpu=1.3
conda activate eqip
conda install -c conda-forge -c cpape pip z5py
conda install -c conda-forge scikit-image numpy scipy
conda install -c conda-forge h5py requests urllib3 idna
conda install -c conda-forge cython
conda install -c conda-forge pymongo
pip install malis==1.0
pip install git+https://github.com/funkey/augment@4a42b01ccad7607b47a1096e904220729dbcb80a
pip install git+https://github.com/funkey/gunpowder@4b34d827f89573b9be10687c66fbeab0836b3fca
pip install git+https://github.com/funkelab/daisy@41130e58582ae05d01d26261786de0cbafaa6482
pip install git+https://github.com/hanslovsky/gunpowder-nodes@7f21b7aa02ba2756e6a6d04c5d4e89bfb1e1196a
pip install git+https://github.com/hanslovsky/eqip@abe20841ccac0bd4badddc5fa8b2f73ae2f94afb
```

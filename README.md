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
pip install git+https://github.com/funkey/gunpowder@06dfc8f5527775e2dc741fd0f33898e3d02bc2c2
pip install git+https://github.com/funkelab/daisy@74b735465954fb01e759d187785ba140e7230f5e
pip install git+https://github.com/hanslovsky/gunpowder-nodes@54bfa540f14cf384472ccfd7e40c481ff5b170b5
pip install git+https://github.com/hanslovsky/eqip@abe20841ccac0bd4badddc5fa8b2f73ae2f94afb
```

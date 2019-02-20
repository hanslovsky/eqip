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
pip install git+https://github.com/hanslovsky/eqip
```

# High Memory Usage/Unexpectedly Many Threads/Processes

[Set environment, as needed](https://stackoverflow.com/a/53224849/1725687):
```sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
 
 [numpy/nump#11826](https://github.com/numpy/numpy/issues/11826)
 

Download Halide lib and extract: 
```
wget https://github.com/halide/Halide/releases/download/v10.0.0/Halide-10.0.0-x86-64-linux-db901f7f7084025abc3cbb9d17b0f2d3f1745900.tar.gz
```

Add to ~/.bshrc
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/creotiv/work/libs/Halide-10.0.0-x86-64-linux/lib
export HALIDE_DIR=/home/creotiv/work/libs/Halide-10.0.0-x86-64-linux
```
Run 
```
python setup.py build
TORCH_USE_RTLD_GLOBAL=YES python3 test.py 
```

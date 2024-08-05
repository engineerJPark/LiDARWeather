<img src="../docs/figs/logo.png" align="right" width="20%">

# Installation

### General Requirement

This codebase is tested with `torch==1.10.0`, `torchvision==0.11.0`, `mmcv==2.0.0rc4`, `mmdet3d==1.2.0`, and `mmengine==0.8.4`, with `CUDA 11.3`. In order to successfully reproduce the results reported in our paper, we recommend you follow the exact same configuration with us.

### Range View
- For the **range view option**, we use [CENet](https://github.com/huixiancheng/CENet) as the LiDAR segmentation backbone. Check out their [original paper](https://arxiv.org/abs/2207.12691) if you need. We set resolutions of the rasterized range image are set as `64x512` for fast training.

### Voxel View
- For the **voxel option**, We support two mainstream voxel-based LiDAR segmentation backbones, i.e., [MinkowskiUNet](https://github.com/NVIDIA/MinkowskiEngine) and [SPVCNN](https://arxiv.org/pdf/2007.16100). 

<hr>

## One-shot Installation
```Shell
conda create -n lidar_weather python=3.8 -y && conda activate lidar_weather
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install -U openmim && mim install mmengine && mim install 'mmcv>=2.0.0rc4, <2.1.0' && mim install 'mmdet>=3.0.0, <3.2.0'

git clone https://github.com/engineerJPark/LiDARWeather.git
cd LiDARWeather && pip install -v -e .

pip install cumm-cu113 && pip install spconv-cu113
sudo apt-get install libsparsehash-dev
export PATH=/usr/local/cuda/bin:$PATH && pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install nuscenes-devkit
pip install wandb
```

<hr>

## Step by Step Installation of [Official Documents](https://mmdetection3d.readthedocs.io/en/latest/)

### Step 1: Create Environment
```Shell
conda create -n lidar_weather python=3.8
```

### Step 2: Activate Environment
```Shell
conda activate lidar_weather
```

### Step 3: Install PyTorch
```Shell
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -y
```

### Step 4: Install MMDetection3D
- **Step 4.1:** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), and [MMDetection](https://github.com/open-mmlab/mmdetection) using [MIM](https://github.com/open-mmlab/mim)
  ```Shell
  pip install -U openmim
  mim install mmengine
  mim install 'mmcv>=2.0.0rc4'
  mim install 'mmdet>=3.0.0'
  ```
  **Note:** In MMCV-v2.x, `mmcv-full` is renamed to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc4"` to install the lite version.

- **Step 4.2:** Install MMDetection3D
  - **Option One:** If you develop and run `mmdet3d` directly, install it from the source:
    ```Shell
    git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
    ```
    **Note:** `"-b dev-1.x"` means checkout to the `dev-1.x` branch.
    
    ```Shell
    cd mmdetection3d
    pip install -v -e .
    ```
    **Note:** `"-v"` means verbose, or more output, `"-e"` means installing a project in editable mode, thus any local modifications made to the code will take effect without reinstallation.

  - **Option Two:** If you use `mmdet3d` as a dependency or third-party package, install it with [MIM](https://github.com/open-mmlab/mim):
    ```Shell
    mim install "mmdet3d>=1.1.0"
    ```

### Step 5: Install Sparse Convolution Backend

- **Step 5.1:** Install SPConv
  - We have supported `spconv 2.0`. If the user has installed `spconv 2.0`, the code will use `spconv 2.0` by default, which will take up less GPU memory than using the default `mmcv` version `spconv`. Users can use the following commands to install `spconv 2.0`:
    ```Shell
    pip install cumm-cuxxx
    pip install spconv-cuxxx
    ```
    Where `xxx` is the CUDA version in the environment. For example, using CUDA 11.3, the command will be `pip install cumm-cu113 && pip install spconv-cu113`.
  
  - The supported CUDA versions include `10.2`, `11.1`, `11.3`, and `11.4`. Users can also install it by building from the source. For more details please refer to [spconv v2.x](https://github.com/traveller59/spconv).


- **Step 5.2:** Install TorchSparse
  - If necessary, follow the [original installation guide](https://github.com/mit-han-lab/torchsparse#installation) or use pip to install it:
    ```Shell
    sudo apt-get install libsparsehash-dev
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    ```
    
  - Or omit sudo install by following command:
    ```Shell
    conda install -c bioconda sparsehash
    export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
    # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    ```

- **Step 5.3:** Install Minkowski Engine (Optional)
  - We also support the Minkowski Engine as a sparse convolution backend. If necessary, follow the [original installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation) or use pip to install it:
    ```Shell
    conda install openblas-devel -c anaconda
    export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
    # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
    ```


### Step 6: Install nuScenes Devkit
:oncoming_automobile: The [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) is **required** in order to run experiments on the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.
```Shell
pip install nuscenes-devkit 
```

### Step 6: Install [WandB](https://wandb.ai/site)
Installing [WandB](https://wandb.ai/site) is optional for training monitoring.
```Shell
pip install wandb
```

## Environment Summary

We provide the list of all packages and their corresponding versions installed in this codebase:
```Shell
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main        
_openmp_mutex             5.1                       1_gnu        
absl-py                   2.0.0                    pypi_0    pypi
addict                    2.4.0                    pypi_0    pypi
aliyun-python-sdk-core    2.14.0                   pypi_0    pypi
aliyun-python-sdk-kms     2.16.2                   pypi_0    pypi
ansi2html                 1.8.0                    pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
asttokens                 2.4.1                    pypi_0    pypi
attrs                     23.1.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
black                     23.11.0                  pypi_0    pypi
blas                      1.0                         mkl        
blinker                   1.7.0                    pypi_0    pypi
bzip2                     1.0.8                h7b6447c_0        
ca-certificates           2023.12.12           h06a4308_0        
cachetools                5.3.2                    pypi_0    pypi
ccimport                  0.4.2                    pypi_0    pypi
certifi                   2023.11.17               pypi_0    pypi
cffi                      1.16.0                   pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi                                                                                                
comm                      0.2.0                    pypi_0    pypi   
configargparse            1.7                      pypi_0    pypi
contourpy                 1.1.1                    pypi_0    pypi
crcmod                    1.7                      pypi_0    pypi
cryptography              41.0.7                   pypi_0    pypi
cudatoolkit               11.3.1               h2bc3f7f_2        
cumm-cu113                0.4.11                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
dash                      2.14.2                   pypi_0    pypi
dash-core-components      2.0.0                    pypi_0    pypi
dash-html-components      2.0.0                    pypi_0    pypi
dash-table                5.0.0                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
descartes                 1.1.0                    pypi_0    pypi
docker-pycreds            0.4.0                    pypi_0    pypi
et-xmlfile                1.1.0                    pypi_0    pypi
exceptiongroup            1.2.0                    pypi_0    pypi
executing                 2.0.1                    pypi_0    pypi
fastjsonschema            2.19.0                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
fire                      0.5.0                    pypi_0    pypi
flake8                    6.1.0                    pypi_0    pypi
flask                     3.0.0                    pypi_0    pypi     
fonttools                 4.46.0                   pypi_0    pypi
freetype                  2.12.1               h4a9f257_0        
giflib                    5.2.1                h5eee18b_3        
gitdb                     4.0.11                   pypi_0    pypi
gitpython                 3.1.40                   pypi_0    pypi
gmp                       6.2.1                h295c915_3        
gnutls                    3.6.15               he1e5248_0        
google-auth               2.24.0                   pypi_0    pypi
google-auth-oauthlib      1.0.0                    pypi_0    pypi
grpcio                    1.59.3                   pypi_0    pypi
idna                      3.6                      pypi_0    pypi
imageio                   2.33.0                   pypi_0    pypi
importlib-metadata        7.0.0                    pypi_0    pypi
importlib-resources       6.1.1                    pypi_0    pypi   
iniconfig                 2.0.0                    pypi_0    pypi
intel-openmp              2023.1.0         hdb19cb5_46306        
ipython                   8.12.3                   pypi_0    pypi
ipywidgets                8.1.1                    pypi_0    pypi
itsdangerous              2.1.2                    pypi_0    pypi
jedi                      0.19.1                   pypi_0    pypi
jinja2                    3.1.2                    pypi_0    pypi
jmespath                  0.10.0                   pypi_0    pypi
joblib                    1.3.2                    pypi_0    pypi
jpeg                      9e                   h5eee18b_1        
jsonschema                4.20.0                   pypi_0    pypi
jsonschema-specifications 2023.11.2                pypi_0    pypi
jupyter-core              5.5.0                    pypi_0    pypi
jupyterlab-widgets        3.0.9                    pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
lame                      3.100                h7b6447c_0        
lark                      1.1.8                    pypi_0    pypi
lazy-loader               0.3                      pypi_0    pypi
lcms2                     2.12                 h3be6417_0         
ld_impl_linux-64          2.38                 h1181459_1                       
lerc                      3.0                  h295c915_0        
libdeflate                1.17                 h5eee18b_1        
libffi                    3.4.4                h6a678d5_0         
libgcc-ng                 11.2.0               h1234567_1        
libgomp                   11.2.0               h1234567_1        
libiconv                  1.16                 h7f8727e_2        
libidn2                   2.3.4                h5eee18b_0        
libpng                    1.6.39               h5eee18b_0        
libstdcxx-ng              11.2.0               h1234567_1        
libtasn1                  4.19.0               h5eee18b_0        
libtiff                   4.5.1                h6a678d5_0        
libunistring              0.9.10               h27cfd23_0        
libuv                     1.44.2               h5eee18b_0        
libwebp                   1.3.2                h11a3e52_0        
libwebp-base              1.3.2                h5eee18b_0        
llvmlite                  0.41.1                   pypi_0    pypi
lyft-dataset-sdk          0.0.8                    pypi_0    pypi
lz4-c                     1.9.4                h6a678d5_0        
markdown                  3.5.1                    pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.3                    pypi_0    pypi
matplotlib                3.5.3                    pypi_0    pypi
matplotlib-inline         0.1.6                    pypi_0    pypi
mccabe                    0.7.0                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mkl                       2023.1.0         h213fc3f_46344                                                                                                        
mkl-service               2.4.0            py38h5eee18b_1           
mkl_fft                   1.3.8            py38h5eee18b_0        
mkl_random                1.2.4            py38hdb19cb5_0        
mmcv                      2.0.1                    pypi_0    pypi
mmdet                     3.1.0                    pypi_0    pypi
mmdet3d                   1.2.0                     dev_0    <develop>
mmengine                  0.10.1                   pypi_0    pypi
model-index               0.1.11                   pypi_0    pypi
mypy-extensions           1.0.0                    pypi_0    pypi
nbformat                  5.7.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0        
nest-asyncio              1.5.8                    pypi_0    pypi
nettle                    3.7.3                hbbd107a_1        
networkx                  3.1                      pypi_0    pypi
ninja                     1.11.1.1                 pypi_0    pypi
numba                     0.58.1                   pypi_0    pypi
numpy                     1.24.3           py38hf6e8229_1        
numpy-base                1.24.3           py38h060ed82_1        
nuscenes-devkit           1.1.11                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi   
open3d                    0.17.0                   pypi_0    pypi
opencv-python             4.8.1.78                 pypi_0    pypi
opendatalab               0.0.10                   pypi_0    pypi     
openh264                  2.1.1                h4ff587b_0        
openjpeg                  2.4.0                h3ad879b_0        
openmim                   0.3.9                    pypi_0    pypi
openpyxl                  3.1.2                    pypi_0    pypi
openssl                   3.0.12               h7f8727e_0        
openxlab                  0.0.29                   pypi_0    pypi
ordered-set               4.1.0                    pypi_0    pypi
oss2                      2.17.0                   pypi_0    pypi
packaging                 23.2                     pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
parso                     0.8.3                    pypi_0    pypi
pathspec                  0.11.2                   pypi_0    pypi
pccm                      0.4.11                   pypi_0    pypi
pexpect                   4.9.0                    pypi_0    pypi   
pickleshare               0.7.5                    pypi_0    pypi
pillow                    10.0.1           py38ha6cbd5a_0        
pip                       23.3.1           py38h06a4308_0        
pkgutil-resolve-name      1.3.10                   pypi_0    pypi
platformdirs              4.1.0                    pypi_0    pypi
plotly                    5.18.0                   pypi_0    pypi
pluggy                    1.3.0                    pypi_0    pypi
plyfile                   1.0.2                    pypi_0    pypi
portalocker               2.8.2                    pypi_0    pypi
prompt-toolkit            3.0.41                   pypi_0    pypi
protobuf                  4.25.1                   pypi_0    pypi
psutil                    5.9.6                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.2                    pypi_0    pypi
pyasn1                    0.5.1                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pybind11                  2.11.1                   pypi_0    pypi
pycocotools               2.0.7                    pypi_0    pypi
pycodestyle               2.11.1                   pypi_0    pypi
pycparser                 2.21                     pypi_0    pypi
pycryptodome              3.19.0                   pypi_0    pypi
pyflakes                  3.1.0                    pypi_0    pypi
pygments                  2.17.2                   pypi_0    pypi
pyparsing                 3.1.1                    pypi_0    pypi
pyquaternion              0.9.9                    pypi_0    pypi
pytest                    7.4.3                    pypi_0    pypi
python                    3.8.18               h955ad1f_0        
python-dateutil           2.8.2                    pypi_0    pypi
pytorch                   1.10.0          py3.8_cuda11.3_cudnn8.2.0_0    pytorch                                                                                 
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.3.post1             pypi_0    pypi
pywavelets                1.4.1                    pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
readline                  8.2                  h5eee18b_0        
referencing               0.31.1                   pypi_0    pypi     
requests                  2.28.2                   pypi_0    pypi
requests-oauthlib         1.3.1                    pypi_0    pypi
retrying                  1.3.4                    pypi_0    pypi
rich                      13.4.2                   pypi_0    pypi
rpds-py                   0.13.2                   pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
scikit-image              0.21.0                   pypi_0    pypi
scikit-learn              1.3.2                    pypi_0    pypi
scipy                     1.10.1                   pypi_0    pypi
sentry-sdk                1.38.0                   pypi_0    pypi
setproctitle              1.3.3                    pypi_0    pypi
setuptools                60.2.0                   pypi_0    pypi
shapely                   1.8.5.post1              pypi_0    pypi
six                       1.16.0                   pypi_0    pypi   
smmap                     5.0.1                    pypi_0    pypi
spconv-cu113              2.3.6                    pypi_0    pypi
sqlite                    3.41.2               h5eee18b_0             
stack-data                0.6.3                    pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0        
tenacity                  8.2.3                    pypi_0    pypi
tensorboard               2.14.0                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
termcolor                 2.4.0                    pypi_0    pypi
terminaltables            3.1.10                   pypi_0    pypi
threadpoolctl             3.2.0                    pypi_0    pypi
tifffile                  2023.7.10                pypi_0    pypi
tk                        8.6.12               h1ccaba5_0        
tomli                     2.0.1                    pypi_0    pypi
torchsparse               1.4.0                    pypi_0    pypi
torchvision               0.11.0               py38_cu113    pytorch
tqdm                      4.65.2                   pypi_0    pypi
traitlets                 5.14.0                   pypi_0    pypi
trimesh                   4.0.5                    pypi_0    pypi
typing_extensions         4.7.1            py38h06a4308_0        
tzdata                    2023.3                   pypi_0    pypi
urllib3                   1.26.18                  pypi_0    pypi
wandb                     0.16.0                   pypi_0    pypi
wcwidth                   0.2.12                   pypi_0    pypi
werkzeug                  3.0.1                    pypi_0    pypi
wheel                     0.41.2           py38h06a4308_0        
widgetsnbextension        4.0.9                    pypi_0    pypi
xz                        5.4.5                h5eee18b_0        
yapf                      0.40.2                   pypi_0    pypi
zipp                      3.17.0                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_0        
zstd                      1.5.5                hc292b87_0
```

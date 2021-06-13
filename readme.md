`vision_transformer` contains a fork of <https://github.com/google-research/vision_transformer>.

### Installing on Linux

From this `notebooks/` directory, run

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install --upgrade jax jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e ../vision_transformer/
pip install -e ../
```

### WSL

System setup:

* WSL 2 or Ubuntu 18.04
* Python v3.8.5
* CUDA v11.0
  * Installation guide for WSL: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>.
* cuDNN v8.0.5
  * Installation guide: <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>.

Follow the [installation guide for CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Then, run the Linux installation instructions.

Note: WSL file I/O is slow.

### Building for `jaxlib` on Windows without WSL

System setup:

* Windows 10 Version 2004 (Build 21296.1010)
* Python v3.8.7 (not from Windows Store, in order to load DLLs from PATH)
* CUDA v11.0
* cuDNN v8.0.5
* GTX 1080 Ti (compute capability 6.1)
* NVIDIA Game Ready Driver 461.40

Follow <https://jax.readthedocs.io/en/latest/developer.html>.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install wheel numpy scipy six
cd ..\jax
python .\build\build.py --enable_cuda --cuda_path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0" --cudnn_path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0" --cuda_compute_capabilities="5.0" --cuda_version="11.0" --cudnn_version="8.0.5"
cd ..\notebooks
pip install ..\jax\dist\jaxlib-0.1.60-cp38-none-win_amd64.whl -e ..\jax
pip install -r requirements.txt
pip install -e ..\vision_transformer -e ..
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

GPU:

```cmd
python -m pip install --upgrade pip
pip install wheel
pip install numpy==1.20 scipy six wheels\jax-3.8.10-11.0-8.0.5\jaxlib-0.1.60-cp38-none-win_amd64.whl -e jax -r requirements.txt -e vision_transformer -e robustness -e . torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

CPU:

```cmd
python -m pip install --upgrade pip
pip install wheel
pip install numpy==1.20 scipy six wheels\jax-3.8.10-11.0-8.0.5\jaxlib-0.1.60-cp38-none-win_amd64.whl -e jax -r requirements.txt -e vision_transformer -e robustness -e . torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

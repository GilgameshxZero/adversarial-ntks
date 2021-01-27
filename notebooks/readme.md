System requirements.

* WSL 2 or Ubuntu 18.04
* python v3.8.5
* CUDA v11.0
  * Installation guide for WSL: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>.
* cuDNN v8.0.5
  * Installation guide: <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>.

Follow the [installation guide for CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel jax jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e vision_transformer
```

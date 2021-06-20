
Please use the below commandlines to clone, install the requirements and load the Conda environment (Note that Cuda 10 is required):

```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```

```bash
git clone https://github.com/microsoft/DialoGPT.git
cd DialoGPT
conda env create -f LSP-linux.yml -n LSP
conda activate LSP
```

If you run this on an architecture other than Linux, please use `LSP-generic.yml` instead of `LSP-linux.yml` but please note that the generic one is not tested in all platform, so the stablity can not be gauranteed.
To use fp16 training, please install apex by using commands below

```bash
conda activate LSP
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard 3d01e4a0a188cc8df54bc6e44cf5eb40ff6b4cc5
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
python3.6 demo.py
```

To start, first install the docker and Nvidia-docker from their official repos.
The image environment for running the code can be loaded as below:  

*Nvidia-docker v2.**

```bash
$ docker run --gpus all --ipc=host --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:dialogpt bash
```
*Nvidia-docker v1.**

```bash
$ nvidia-docker --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:dialogpt bash
```

Inside the docker container, run 

```bash
python gdpt_demo.py
```

This section explains all components in the `demo.py`.

Before running `gdpt_demo.py`, you can set *DATA_FOLDER* (default value `./models`)  in `gdpt_demo.py` as the place you want to download all the data and pretrained/fine-tuned models. Then simply run
```bash
python gdpt_demo.py
```
to generate a training scripts.

The pretrained and fine-tuned models are available on azure blobstorage.
Please run/see `gdpt_demo.py` for more details about how to download/use those models. Or you could download directly by using the links in `demo_utils.py`.

First, use the `prepare4db.sh` to convert a tsv data file into the correct format that the following script can recognize.
The trainig data need to be then processed into a database file with below commandline:

```bash
python prepro.py --corpus $DATA_PATH
```

The training script can be used in single GPU or multiple GPU settings (distributed training across multiple GPUs within a single node):

```bash
python ./LSP_train.py  # Single GPU training
python -m torch.distributed.launch --nproc_per_node=8 ./LSP_train.py  # Training on 8 GPUs
```

The training script accept several arguments to tweak the training: 

Argument | Type | Default value | Description
---------|------|---------------|------------
max\_seq\_length | `int` | `128` | Maximum number of tokens for each training instance. 
train\_input\_file | `str` | `""` | Path of the training dataset in a .db format
eval\_input\_file | `str` | `""` | Path of the validation set in a tsv format
continue_from | `int` | `0` | Resuming the training after a specified number of steps
fp16 | `boolean` | `True` | Whether to use 16-bits floating point for model training.
train\_batch\_size | `int` | `4` | Batch size for training
valid\_batch\_size | `int` | `4` | Batch size for validation
gradient\_accumulation\_steps | `int` | `2` | Accumulate gradients on several steps
learning\_rate | `float` | `1e-5` | Learning rate
lr\_schedule | `str` | `noam` | Learning rate schedule can be chosen from [`noam`, `noamwd`, `BERT`, `None`]
num\_optim\_steps | `int` | `1000000` | Number of training optimization steps
no_token_id | `boolean` | `True` | If set True, using all-zeros token-type embedding.

During the training, two log files will be updated. The `train_log.txt` and `eval_log.txt` contains the model loss, perplexity and training speed (tokens/sec) statistics for the training and dev set. 

The log file and saved model checkpoint can be found in `./models/output_model`

We release 6 fine-tuned models which can be further fine-tuned on low-resource  user-customized dataset. The total parameters in these models range from 117M to 762M, in accord with OpenAI GPT-2 model sizes.   

| Model           |  Download|
|----------------------|--------|
| DialoGPT 762M model| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/large_ft.pkl) |
| DialoGPT 345M model| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl) |
| DialoGPT 117M model| [link](https://convaisharables.blob.core.windows.net/lsp/multiref/small_ft.pkl) |

The model files can be loaded exactly as the GPT-2 model checkpoint from Huggingface [pytorch-transformer](https://github.com/huggingface/transformers). Please download the required model configuration files (`merges.txt`, `config,json`, `vocab.json`) from `./configs/*`.

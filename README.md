# RetGen: A Joint framework for Retrieval and Grounded Text Generation Modeling

This repository contains the source code and trained model for ["Joint Retrieval and Generation Training for Grounded Text Generation"](https://arxiv.org/abs/2105.06597). RetGen is a joint training framework that simultaneously optimizes a dense passage retriever and a knowledge-grounded text generator in an end-to-end fashion. It can be applied to scenarios including but not limited to conversational modeling, text generation and open-domain question answering. The code implementation is based on [DialoGPT](https://github.com/microsoft/DialoGPT), [Huggingface Transformers](https://github.com/huggingface/transformers), [DPR](https://github.com/facebookresearch/DPR) and [ANCE](https://github.com/microsoft/ANCE). Our [human evaluation results](#human_eval) indicates that RetGen can generate more relevant, interesting and human-like text comparing to vanilla DialoGPT or GPT-2.


![Screenshot](figs/model.pdf) Figure: RetGen overview.


## Citation
If this repo is helpful to your research, please cite our [paper](https://arxiv.org/abs/2105.06597):
```bash
@article{zhang2021joint,
  title={Joint Retrieval and Generation Training for Grounded Text Generation},
  author={Zhang, Yizhe and Sun, Siqi and Gao, Xiang and Fang, Yuwei and Brockett, Chris and Galley, Michel and Gao, Jianfeng and Dolan, Bill},
  journal={arXiv preprint arXiv:2105.06597},
  year={2021}
}
```


## Enviroment
### Conda

For cuda 10.0, run
```bash
conda env create -f RetGen.yml
conda activate RetGen
conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch
```

, then install apex by  (download apex to somewhere else)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
For cuda 10.1, simply run
```bash
conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
```
instead of
```bash
conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch
```

Next, install Fairseq in somewhere else
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### Docker

Container was built by
```bash
docker build -f dockerfile.txt -t gdpt .
```
Activate container by 
```bash
docker run --gpus all --ipc=host --rm -it --mount src=/your_source_code_dir,dst=/code,type=bind --mount src=/gdpt,dst=/gdpt,type=bind intersun/gdpt
```

### Preprocessing
The training corpus needs to be first compressed in to *.db file using command below
```python
python dialogpt/prepro.py --corpus data/train.doc_ctx_rsp.txt --max_seq_len 512
``` 

### Training
Example training command for reddit data with 8 GPUs (each with 32GB VRAM)

```python
python -m torch.distributed.launch --nproc_per_node=8 joint_training.py
     --model_name_or_path configs
     --init_checkpoint models/reddit_generator.pkl
     --train_input_file data/reddit_train.db
     --eval_input_file data/reddit_test.txt 
     --output_dir output/joint_reddit
     --file_suffix joint_reddit
     --train_batch_size 4
     --gradient_accumulation_steps 2
     --eval_batch_size 2
     --num_optim_steps 16000
     --encoder_model_type ance_roberta
     --pretrained_model_cfg bert-base-uncased
     --model_file models/reddit_retriever.pkl
     --ctx_file data/wiki.txt
     --num_shards 8
     --batch_size 128
     --n_docs 4
     --encoding
     --load_trained_model
```
For single card, please use following command

```python
CUDA_VISIBLE_DEVICES=0 python joint_training.py
     --model_name_or_path configs
     --init_checkpoint models/reddit_generator.pkl
     --train_input_file data/reddit_train.db
     --eval_input_file data/reddit_test.txt 
     --output_dir output/joint_reddit
     --file_suffix joint_reddit
     --train_batch_size 2
     --gradient_accumulation_steps 2
     --eval_batch_size 2
     --num_optim_steps 16000
     --encoder_model_type ance_roberta
     --pretrained_model_cfg bert-base-uncased
     --model_file models/reddit_retriever.pkl
     --ctx_file data/wiki.txt
     --num_shards 1
     --batch_size 128
     --n_docs 2
     --encoding
     --load_trained_model
```

### Model checkpoints


We release model checkpoints which can be directly used or further fine-tuned with customized dataset. 

| Model           |  Reddit| arXiv
|----------------------|--------|--------|
| RetGen generator| [\[link\]](https://yizzhang.blob.core.windows.net/gdpt/RetGen/models/reddit_generator.pkl?sv=2020-04-08&st=2021-06-15T18%3A36%3A51Z&se=2025-06-16T18%3A36%3A00Z&sr=b&sp=r&sig=2hHNK2b%2BnyLd69iLLHd%2BhlC1fyu2eTODxYGHPJFkSHc%3D)| [\[link\]](https://yizzhang.blob.core.windows.net/gdpt/RetGen/models/arxiv_generator.pkl?sv=2020-04-08&st=2021-06-15T18%3A36%3A22Z&se=2025-06-16T18%3A36%3A00Z&sr=b&sp=r&sig=KyzO6Jt7wNUvKPOhJduFKqVsacM6ytW38xQsxsOYvWQ%3D) |
| RetGen retriever| [\[link\]](https://yizzhang.blob.core.windows.net/gdpt/RetGen/models/reddit_retriever.pkl?sv=2020-04-08&st=2021-06-15T18%3A37%3A13Z&se=2021-06-16T18%3A37%3A13Z&sr=b&sp=r&sig=EWHcFvFloZMuNtoH0nnsbf6Ed8D7b5EQNv1Qe68rO8M%3D)| [\[link\]](https://yizzhang.blob.core.windows.net/gdpt/RetGen/models/arxiv_retriever.pkl?sv=2020-04-08&st=2021-06-15T18%3A37%3A33Z&se=2025-06-16T18%3A37%3A00Z&sr=b&sp=r&sig=m51gD41OynhkKylP0ysYrL6bL9bqJ2j7f8HXwbUYy3c%3D) | 


For the generator, you can find the corresponding configuration files (`merges.txt`, `config.json`, `vocab.json`) in `./configs/*`.


### Data

The preprocessed wikipedia dump used in our work can be downloaded as in below. We also provide the raw text data for arXiv. For Reddit, due to copyright issue. We will release a script to automatically extract the training development and test data. 

| Data          |  Link |
|----------------------|--------|
| Wiki (2.5GB)| [\[link\]](https://yizzhang.blob.core.windows.net/gdpt/RetGen/data/wiki.txt?sv=2020-04-08&st=2021-06-15T18%3A32%3A15Z&se=2025-01-16T19%3A32%3A00Z&sr=b&sp=r&sig=LPBAKDIRx7UYPuGuvfWKQPe8URbMGpuHnXwwb7gv5mw%3D)| N/A | N/A |
| arXiv (2.4GB)| [\[link\]](https://yizzhang.blob.core.windows.net/gdpt/RetGen/data/arxiv.tar.gz?sv=2020-04-08&st=2021-06-15T18%3A34%3A01Z&se=2025-01-06T19%3A49%3A00Z&sr=b&sp=r&sig=gYiL2FQWwNYmEo2823QNnkmPMiqD2I7CvyHc%2Fg8o1fs%3D)| 
| Reddit| TBD  |




### Inference
We note that even with properly filtered Reddit dataset, sometimes our model can still generate moderately toxic/inappropriate responses. Due to this reason, we are unable to provide the inference code at this time. We are currently working on a controlled decoding method to prevent this system from toxic generation. Please stay tuned.

### Evaluation
#### Generation Evaluation

Please follow [dialoGPT](https://github.com/microsoft/DialoGPT) evaluation script in dialogpt/README.md. 

#### Retrieval Evaluation
We provide example for evaluation retriever. The evaluation is done by estimating the recall@10,30,50 over 10K samples. The required files can be downloaded [here](https://yizzhang.blob.core.windows.net/gdpt/RetGen/data/ret_eval.tar.gz?sv=2020-04-08&st=2021-06-15T22%3A43%3A06Z&se=2025-06-16T22%3A43%3A00Z&sr=b&sp=r&sig=S9WHqJJsYPpY7KXcvfYj36XP4EfcEuCIhDYVxVWcgVc%3D)

```
CUDA_VISIBLE_DEVICES=0 python eval_checkpoint.py \
        --eval_mode rank \
        --encoder_model_type ance_roberta \
        --pretrained_model_cfg bert-base-uncased \
        --model_file models/reddit_retriever.pkl \
        --qa_file data/2k_positive.txt \
        --ctx_file data/10k.txt \
        --n_docs 50 \
        --batch_size 64 \
        --shard_id 0 \
        --num_shards 1 \
        --load_trained_model \
        --encoding
```

If running correctly, the system should output results as in below:
```
Validation results: recall@10:0.617
Validation results: recall@30:0.703
Validation results: recall@50:0.744
```

### <a name="human_eval"></a>Human evaluation 

We further conduct human evaluations (500 examples for each methods, each example is evaluated by 3 human judges). The results show a strong evidence that our generation quality is better than vanilla DialoGPT/GPT-2, under this non-interactive Turing test:


*Coherence*:A and B, which is more relevant to, and coherent with the context?

|Dataset | System A | A Wins (%) | Ties (%) | B Wins (%) | System B|
|--------|--------------------|-------|-------|--------|-------|
|Reddit  |RetGen 345M|43.7   | 28.3 |   28.0| DialoGPT 345M|
|arXiv   |RetGen 345M|32.1   | 41.7 |   26.3| GPT-2 345M|


*Informativeness*: A and B, which is more informative (usually more specific content)?
|Dataset | System A | A Wins (%) | Ties (%) | B Wins (%) | System B|
|--------|--------------------|-------|-------|--------|-------|
|Reddit  |RetGen 345M|44.5   | 27.8 |   27.7| DialoGPT 345M|
|arXiv   |RetGen 345M|36.3   | 37.2 |   26.5| GPT-2 345M|

*Human-likeness*: A and B, which is more likely to be generated by human rather than a machine?
|Dataset | System A | A Wins (%) | Ties (%) | B Wins (%) | System B|
|--------|--------------------|-------|-------|--------|-------|
|Reddit  |RetGen 345M|36.4   | 34.0 |   29.6| DialoGPT 345M|
|arXiv   |RetGen 345M|29.7   | 43.6 |   26.7| GPT-2 345M|

## Related Project

* DialoGPT: [https://github.com/microsoft/DialoGPT](https://github.com/microsoft/DialoGPT). 
A State-of-the-Art Large-scale Pretrained Response Generation Model

## Contact

Please contact [DialoGPT@microsoft.com](mailto:DialoGPT@microsoft.com) if you have any questions/suggestions. However, the response will be sporadic. Please expect delay.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. 

## Disclaimer

This repository aims to facilitate research in large-scale pretraining for conversational data. This toolkit contains only part of the modeling machinery needed to actually produce a model weight file in a running dialog. On its own, this model provides only information about the weights of various text spans; in order for a researcher to actually use it, they will need to bring conversational data of their own and decode the response generation from the pretrained system. We are not responsible for any generation from the 3rd party utilization of the pretrained system. 



# RetGen: A Joint framework for Retrieval and Grounded Text Generation Modeling

This repository contains the source code and trained model for ["Joint Retrieval and Generation Training for Grounded Text Generation"](https://arxiv.org/abs/2105.06597). RetGen is a joint training framework that simultaneously optimizes a dense passage retriever and a knowledge-grounded text generator in an end-to-end fashion. It can be applied to scenarios including but not limited to conversational modeling, text generation and open-domain question answering. 

Code to be updated soon, please stay tuned.


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

```python
python prepro.py --corpus data/train.doc_ctx_rsp.txt --max_seq_len 512
``` 

### Training
Example training command for reddit data with 8 GPUs

```python
python -m torch.distributed.launch --nproc_per_node=8 joint_training.py
     --model_name_or_path models/medium
     --init_checkpoint models/GP2-pretrain-step-0.pkl
     --train_input_file data/train.doc_ctx_rsp.512len.db
     --eval_input_file data/multiref.doc_ctx_rsp.txt 
     --output_dir output/joint_reddit
     --file_suffix joint_reddit
     --train_batch_size 4
     --gradient_accumulation_steps 2
     --eval_batch_size 2
     --learning_rate 1e-6
     --num_optim_steps 16000
     --valid_step 200
     --warmup_steps 10
     --normalize_data True
     --lr_schedule None
     --loss_scale 0.0
     --no_token_id False
     --pbar False
     --set_type_embedding_to_zero True
     --encoder_model_type hf_bert
     --pretrained_model_cfg bert-base-uncased
     --model_file model/Ret-pretrain-step-0.pkl
     --ctx_file data/wikipedia/wikipedia.ten_para.w100.re.txt
     --num_shards 8
     --batch_size 128
     --n_docs 4
     --rl_method simple
```


### Inference
Example inference command for reddit data 

```python
python run_gpt2_multidoc.py
     --use_gpu 
     --gpu 0
     --load_checkpoint models/generator-pretrain-step-{step}.pkl
     --model_name_or_path models/medium
     --generation_length 50
     --is_sampling greedy
     --top_k 0
     -t data/reddit/multiref.doc_ctx_rsp.txt
     -o joint_{step}
     --encoder_model_type ance_roberta
     --pretrained_model_cfg bert-base-uncased
     --model_file models/retriever-pretrain-step-{step}.pkl
     --ctx_file data/wikipedia/wikipedia.ten_para.w100.re.txt
     --shard_id 0 
     --num_shards 1
     --batch_size 128
     --n_docs 4
     --use_own_generation
     --only_print_consensus
```

### Evaluation

Please follow dialoGPT evaluation script in dialogpt/README.md


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


## Citation
If you use this code in your research, you can cite our [paper](https://arxiv.org/abs/2105.06597):
```bash
@article{zhang2021joint,
  title={Joint Retrieval and Generation Training for Grounded Text Generation},
  author={Zhang, Yizhe and Sun, Siqi and Gao, Xiang and Fang, Yuwei and Brockett, Chris and Galley, Michel and Gao, Jianfeng and Dolan, Bill},
  journal={arXiv preprint arXiv:2105.06597},
  year={2021}
}
```
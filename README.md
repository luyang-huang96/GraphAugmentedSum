# Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward
Code for ACL2020 paper: Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward

We are not allowed to share data/outputs on New York Times Dataset. If you need data/outputs on New York Times Dataset, please email me with your license and we're glad to share our processed data/outputs on NYT dataset for research purpose.  

My permenant email address: luyang.huang96@gmail.com  


## How to train our model  

1. our processed data with constructed graphs can be found here:  



2. To train our best model:  

0) specify data path  
`export DATA=[path/to/decompressed/data]`

1) train our model with ML objective

```
python train_abstractor.py --batch 32 --max_input 512 --bert --docgraph(--paragraph for SegGraph extension) --path [path/to/ml/model]
```

2) train our model with our cloze reward

```
python train_abstractor_rl.py --abs_dir [path/to/ml/model] --docgraph(--paragraph for SegGraph extension) --batch 32 --max_art 512 --reward_model_dir [/path/to/reward/model] --reward_data_dir [/path/to/reward/data] --path [/path/to/best/model]
```

3) decode 
```

```

4) evaluate  
```

```





## Dependencies  
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch)
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)
- [transformers]()

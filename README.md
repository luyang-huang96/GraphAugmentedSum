# Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward
Code for ACL2020 paper: Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward

We are not allowed to share data/outputs on New York Times Dataset. If you need data/outputs on New York Times Dataset, please email me with your license and we're glad to share our processed data/outputs on NYT dataset for research purpose.  

My permenant email address: luyang.huang96@gmail.com  


## How to train our model  

I. our processed data with constructed graphs can be found here:  

https://drive.google.com/open?id=1ccja3oyWIJIm91EiG-NJPFNb4Eg1pOmO  

our processed cloze questions can be found here:   

https://drive.google.com/open?id=16aPmfT9Gurjhg1uLeVAUTL7fTc6TO42W  

our best model can be found here:   

https://drive.google.com/open?id=19HeT3rr2mzvEx82arrvpSVOBM_JNeRzo  

our trained cloze model can be found here:   

https://drive.google.com/open?id=1SxIitGBuPmOfKPHQ21LIX_OJ1RUpHpsk  

our best system results/reference can be found here:  

https://drive.google.com/open?id=1SRLCVb-YtCzL5NgI76CXby_Oc_MczYjk  
https://drive.google.com/open?id=1uXn-dyN4KH4LYzKsCDCVnvRDbbR-lAAV  


II. To train our best model:  

0) specify data path  
`export DATA=[path/to/decompressed/data]`

1) train our model with ML objective

```
python train_abstractor.py --batch 32 --max_input 512 --bert --docgraph(--paragraph for SegGraph extension) --path [path/to/ml/model]
```

2) train our model with our cloze reward

```
python train_abstractor_rl.py --abs_dir [path/to/ml/model] --docgraph(--paragraph for SegGraph extension) --batch 32 --max_art 512 --reward_model_dir [/path/to/cloze/model] --reward_data_dir [/path/to/cloze/data/questions] --path [/path/to/best/model]
```

3) decode 
```
python decode_abs.py --abs_dir [/path/to/best/model] --test --reverse --docgraph(--paragraph)  --gpu_id 0 --path [/path/to/results]
```

4) evaluate ROUGE
```
export ROUGE=[/path/to/ROUGE 1.5.5]
```
```
python evaluate_full_model.py --decode_dir [/path/to/results] --rouge
```

5) evaluate QA

```
python eval_cloze_model.py --system_path [/path/to/results] --data_path [/path/to/cloze/data/questions]  --model_dir [/path/to/cloze/model]
```

III. To train our multiple choice QA model  

```
python train_roberta_multiple_choice.py --path [/path/to/cloze/data/training] --save_path [/path/to/cloze/model]
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



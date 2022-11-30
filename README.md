# PVLM (Prompt-based Vision-aware Language Modeling)

This is the implementation of our ICME 2022 paper "[Few-shot Multi-modal Sentiment Analysis with Prompt-based Vision-aware Language Modeling](https://ieeexplore.ieee.org/document/9859654)".

## Requirements

- python 3.8
- pytorch 1.7.1
- timm 0.4.12
- scikit-learn 0.24.2 
- tqdm 4.59.0
- transformers 4.6.0
- pillow 8.2.0
- numpy 1.20.2

we mainly use `conda` commands to set up our environment and only use `pip` for installing `timm`.

## Prepare the data

You can download the datasets from the original links below and preprocess the data following the scrips in `datasets_pre_processing`, or you can download images from our [BaiduNetdisk](https://pan.baidu.com/s/1MBgH2SGg5D3WDepXhD_b4A) (access code `d2w2`) and use preprocessed data in `datasets` directly.

>**Original links**
>
>Twitter-15 and Twitter-17: [A Google Drive link](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view) from https://github.com/jefferyYu/TomBERT
>
>MASAD: [A Google Drive link](https://drive.google.com/file/d/19YJ8vEYCb-uEKUqSGFmysUTvNzxhVKFE/view?usp=sharing) from https://github.com/12190143/MASAD
>
>MVSA-S: [A OneDrive link](https://portland-my.sharepoint.com/:u:/g/personal/shiaizhu2-c_my_cityu_edu_hk/Ebcsf1kUpL9Do_u4UfNh7CgBC19i6ldyYbDZwr6lVbkGQQ) from http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
>
>TumEmo: Two BaiduNetdisk links from https://github.com/YangXiaocui1215/MVAN

## Run PVLM

### Quick start

```shell
python main.py \
    --cuda 0 \
    --out_dir 'out' \
    --dataset 'mvsa-s' \
    --img_dir 'MVSA-S_data' \
    --template 1 \
    --prompt_shape '33-3' \
    --few_shot_file 'few-shot1.tsv' \
    --img_token_len 3 \
    --batch_size 32 \
    --lr_lm_model 1e-5 \
    --lr_visual_encoder 0 \
    --early_stop 40 \
    --seed 5
```

The prediction result will be saved in a text file in path `out/mvsa-s/[s1][t1][nf_resnet50-3]/1e-5/`. 

`[s1]` stands for few-shot1.tsv (the few-shot training file) and `[t1]` stands for template 1. `[nf_resnet50-3]` suggests that we use NF-ResNet50 as the visual encoder (default setting) and set the length of the visual tokens to 3.

We further explain some other arguments:

- `cuda`: The cuda device number, must be one specified integer. 
- `img_dir`: The unziped image directory.
- `prompt_shape`: The number of learnable tokens in each `[L-Tokens]`, only works when using template 3. The integer after `-` means the number of learnable tokens that come with the visual tokens. It should be 0 if we run PT.
- `lr_lm_model`: The learning rate of the language model.
- `lr_visual_encoder`: The learning rate of the visual encoder. In our experiments, we found that a smaller learning rate seems to bring better results. Thus, we set it to 0 for convenience. Of course you can try other values to acquire better performance.

### Experiments with multiple runs

We provide `run/run.sh` to carry out our experiments (with a grid search on `lr_lm_model` and `img_token_len`, running 3 times with different random seeds).

After a simple configuration, you can get all the results on a specific dataset using the specified template. You can try other `dataset` with other `few_shot_file` and `template`, too.

```shell
bash run/run.sh
```

We report the average performance as we mentioned in the [supplementary material](https://ieeexplore.ieee.org/ielx7/9859562/9858923/9859654/976_S.zip). You can obtain the maximum values, mean values and standard deviations according to `gather_results.py`. Do not forget to modify the dataset and template settings in `gather_results.py` before running.

```shell
python gather_results.py
```

The average performance will be stored as a text file in the corresponding directory. 

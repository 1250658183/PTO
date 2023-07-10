# PTO

This the office code for the paper On Prefix-tuning for Lightweight Out-of-distribution Detection.

The paper has been accepted in ACL 2023.

### PTO Group

    cd ./ &&  python trainer_PTO.py --seed 47 --dataset clinc
    
    % Reproducible Seed: CLinc: [47, 7226, 7980, 383, 1948], IMDB: [3669, 2874, 9336, 8148, 2294]

### PTO + Label Group

    cd ./separate_prefix &&  python trainer4sep.py --seed 7760 --dataset clinc

    % Reproducible Seed: CLinc: [7760, 254, 5253, 105, 8187], IMDB: [7005, 9096, 2099, 298, 1850]

### * + OOD Group

    cd ./separate_prefix &&  python tester4_OODprefix.py


### Dataset

The dataset is stored in [Google Drive](https://drive.google.com/file/d/18_eQIC2sWTe7dydYDWOHQLPrk7C0Bjtf/view?usp=sharing), unzip and replace the dataset directory.

## Citation
```bibtex
@inproceedings{ouyang-etal-2023-prefix,
    title = "On Prefix-tuning for Lightweight Out-of-distribution Detection",
    author = "Ouyang, Yawen  and
      Cao, Yongchang  and
      Gao, Yuan  and
      Wu, Zhen  and
      Zhang, Jianbing  and
      Dai, Xinyu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.85",
    pages = "1533--1545",
}
```
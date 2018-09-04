# Visual Module for Communication in Context

Public code for [https://github.com/judithfan/visual_communication_in_context](https://github.com/judithfan/visual_communication_in_context).

#### Setup

Download dependencies:

```
pip install -r ./requirements.txt
```

Download data:
    
    - Go to https://figshare.com/projects/Visual_Communication_in_Context/38321
    - Download all zip files (should be 3) to `./visual_module/data` folder
    - Unzip all 3 files

#### Running files

To train using all 5 splits, run:

```
python run_cv5.py high --out-dir ./trained_models
```

The first positional argument can be `high`, `mid`, or `early` depending on your chosen adaptor type.

To get similarity metrics and prepare for Bayesian data analysis, run:

```
python eval_cv5.py ./trained_models --out-dir ./similarity_dump
```

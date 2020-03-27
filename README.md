# UrcaNet
Source code for Neural Conversational QA: Learning to Reason v.s. Exploiting Patterns (https://arxiv.org/abs/1909.03759)

## Prerequisites

### Clone repository and install AllenNLP

```bash
conda create -n orca python=3.6  
source activate orca  
git clone https://github.com/IBM/UrcaNet.git   
cd UrcaNet  
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Download data
```bash
wget https://sharc-data.github.io/data/sharc1-official.zip     
unzip sharc1-official.zip
```

Also download CoQA and QuAC data and put it in `coqa` and `quac` folder respectively.

## Experiments

## Full Task

### BiDAF baseline

```bash
allennlp train experiments/bidaf_baseline_ft.jsonnet -s ./temp/bidaf_baseline_ft --include-package orca    
python evaluate.py ./temp/bidaf_baseline_ft/model.tar.gz bb_ft
```

### CopyNet on top of BiDAF

```bash
allennlp train experiments/bidaf_copynet_ft.jsonnet -s ./temp/bidaf_copynet_ft --include-package orca    
python evaluate.py ./temp/bidaf_copynet_ft/model.tar.gz bc_ft
```

## Question Generation

Change `task = 'full'` to `task = 'qgen'` in `evaluate.py`.

### BiDAF baseline

```bash
allennlp train experiments/bidaf_baseline.jsonnet -s ./temp/bidaf_baseline --include-package orca    
python evaluate.py ./temp/bidaf_baseline/model.tar.gz bb
```

### CopyNet baseline

```bash
allennlp train experiments/copynet_baseline.jsonnet -s ./temp/copynet_baseline --include-package orca    
python evaluate.py ./temp/copynet_baseline/model.tar.gz cb
```

### CopyNet on top of BiDAF

```bash
allennlp train experiments/bidaf_copynet.jsonnet -s ./temp/bidaf_copynet --include-package orca    
python evaluate.py ./temp/bidaf_copynet/model.tar.gz bc
```

### CopyNet BiDAF Pipeline

```bash
allennlp train experiments/bidaf_copynet_pipeline.jsonnet -s ./temp/bidaf_copynet_pipeline --include-package orca    
python evaluate.py ./temp/bidaf_copynet_pipeline/model.tar.gz bcp
```

## Data Generation

Use `create_new_dataset.py` to create the `ShARC-Augmented` dataset. More details can be found in section 4.1 of the [paper](https://arxiv.org/abs/1909.03759).

## Heuristic Baselines

The rule-based heuristic baseline can be found in `rule.ipynb`. More details can be found in section 4.3 of the [paper](https://arxiv.org/abs/1909.03759).

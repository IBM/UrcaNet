# Prerequisites

## Clone repository and install AllenNLP

> conda create -n orca python=3.6  
> source activate orca  
> git clone https://github.com/IBM/UrcaNet.git   
> cd UrcaNet  
> pip install -r requirements.txt
> python -m spacy download en_core_web_md

## Download data
> wget https://sharc-data.github.io/data/sharc1-official.zip     
> unzip sharc1-official.zip

Also download CoQA and QuAC data and put it in `coqa` and `quac` folder respectively.

# Experiments

## Full Task

### BiDAF baseline

> allennlp train experiments/bidaf_baseline_ft.jsonnet -s ./temp/bidaf_baseline_ft --include-package orca    
> python evaluate.py ./temp/bidaf_baseline_ft/model.tar.gz bb_ft

### CopyNet on top of BiDAF

> allennlp train experiments/bidaf_copynet_ft.jsonnet -s ./temp/bidaf_copynet_ft --include-package orca    
> python evaluate.py ./temp/bidaf_copynet_ft/model.tar.gz bc_ft

## Question Generation

Change `task = 'full'` to `task = 'qgen'` in `evaluate.py`.

### BiDAF baseline

> allennlp train experiments/bidaf_baseline.jsonnet -s ./temp/bidaf_baseline --include-package orca    
> python evaluate.py ./temp/bidaf_baseline/model.tar.gz bb

### CopyNet baseline

> allennlp train experiments/copynet_baseline.jsonnet -s ./temp/copynet_baseline --include-package orca    
> python evaluate.py ./temp/copynet_baseline/model.tar.gz cb

### CopyNet on top of BiDAF

> allennlp train experiments/bidaf_copynet.jsonnet -s ./temp/bidaf_copynet --include-package orca    
> python evaluate.py ./temp/bidaf_copynet/model.tar.gz bc

### CopyNet BiDAF Pipeline

> allennlp train experiments/bidaf_copynet_pipeline.jsonnet -s ./temp/bidaf_copynet_pipeline --include-package orca    
> python evaluate.py ./temp/bidaf_copynet_pipeline/model.tar.gz bcp
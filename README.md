# Prerequisites

## Clone repository and install AllenNLP

> conda create -n orca python=3.6  
> source activate orca  
> git clone https://github.ibm.com/abhis200/ShARC-task.git   
> cd ShARC-task  
> pip install -r requirements.txt

## Download data
> wget https://sharc-data.github.io/data/sharc1-official.zip     
> unzip sharc1-official.zip

# Experiments

## BiDAF baseline

> allennlp train experiments/bidaf_baseline.jsonnet -s ./temp/bidaf_baseline --include-package orca    
> python evaluate.py ./temp/bidaf_baseline/model.tar.gz

## CopyNet baseline

> allennlp train experiments/copynet_baseline.jsonnet -s ./temp/copynet_baseline --include-package orca    
> python evaluate.py ./temp/copynet_baseline/model.tar.gz

## CopyNet on top of BiDAF

> allennlp train experiments/bidaf_copynet.jsonnet -s ./temp/bidaf_copynet --include-package orca    
> python evaluate.py ./temp/bidaf_copynet/model.tar.gz

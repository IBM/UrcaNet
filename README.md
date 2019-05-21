# Install Allennlp

> conda create -n orca python=3.6  
> source activate orca  
> pip install -r requirements.txt

# Download data
> wget https://sharc-data.github.io/data/sharc1-official.zip
> unzip sharc1-official.zip

# BiDAF baseline

> allennlp train experiments/bidaf_baseline.jsonnet -s ./temp/sharc_bidaf --include-package orca  
> python evaluate_bidaf_baseline.py
# TSI-CNet
  

# Prerequisite

- Pytorch
- We provided requirement file to install all packages, just by running


`pip install -r requirements.txt`

 
<a name="Data"></a>

# Data
## Generate the  data 

**Download the raw data**

- [NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/). 

**Preprocess**
- Preprocess data path /data, with `python ntu_gendata.py`.

<a name="Training&Testing"></a>

# Training&Testing
## Training 

- To train on NTU-RGB+D 60 under Cross-Subject evaluation, you can run


    `python ./pretraining.py --lr 0.01 --batch-size 64 --encoder-t 0.2   --encoder-k 8192  
                --schedule 451  --epochs 551  --pre-dataset ntu60 
                --protocol cross_subject --skeleton-representation joint`




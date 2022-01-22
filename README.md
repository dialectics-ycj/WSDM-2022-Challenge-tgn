## Introduction


#### Paper link: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)
#### Code Based on: [TGN](https://github.com/twitter-research/tgn)

## Running the experiments

### Requirements
```{bash}
image: pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
  image_setup:
    - pip install --upgrade pip
    - pip install numpy
    - pip install -U scikit-learn
    - pip install matplotlib
    - pip install pandas
    - pip install requests
```

### Dataset and Preprocessing
Please make sure your current working directory is `$PROJECT/src`
#### Directory initialization 
```{bash}
python dir_init.py
```
`$PROJECT/data`,`$PROJECT/output` will be created to store data, logs and results.
#### Download the data
```{bash}
python utils/getData.py
```
It will download data from WSDM-2022-Challenge, and store them in `$PROJECT/data`.
#### Preprocess the data
```{bash}
python utils/preprocess_data.py --all
```
It will preprocess the data, and store the result in `$PROJECT/data`.

It will also generate a set of random numbers, and output them in `$PROJECT/data/output_A.csv`,`$PROJECT/data/output_B.csv`. These files are used to combine with the **intermediate ** prediction results to form `output.zip`. And `output.zip` will be submit to the  [quick evaluation platform](http://eval-env.eba-5u39qmpg.us-west-2.elasticbeanstalk.com/) automatically to evaluate our model. They will not affect  the **final test**.

### Model Training
Please note that it will generate `$RPOJECT/data/neighborFinder/neighborFineder_$DATA_NAME_full.pth` the **FIRST TIME** you run this code. Next time, these files will be used to accelerate data generation.

**Dataset A:**

```{bash}
/bin/bash remote_run_A.sh
```
By default, the code will select the best model from *10* epochs to generate the **final** prediction result based on the **intermediate** evaluation results. It will cost about 10*4.5h on a P100 GPU.
You can also reduce the number of epoch by change `n_epoch` in the `remote_run_A.sh`.

The code will generate three types results, and output the them to `$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_${RESULT_TYPE}.csv`
```angular2
$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_sum.csv
$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_mean.csv
$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_max.csv
```
We submit `output_${DATA_NAME}_${EPOCH}_max.csv` as the **final** prediction results.

**Dataset B:**

```{bash}
/bin/bash remote_run_B.sh
```
By default, the code will select the best model from *15* epochs to generate the **final** prediction result based on the **intermediate** evaluation results.  It will cost about 15*1.8h on a P100 GPU.
You can also reduce the number of epoch by change `n_epoch` in the `remote_run_B.sh`.

The code will generate three types results, and output the them to `$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_${RESULT_TYPE}.csv`
```angular2
$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_sum.csv
$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_mean.csv
$PROJECT/output/logs/$MODEL_ID/predict-result/output_${DATA_NAME}_${EPOCH}_max.csv
```
We submit `output_${DATA_NAME}_${EPOCH}_max.csv` as the **final** prediction results.






# ADMETNet
A Neural Network Framework for AMDET Prediction. 

**Installation**
```
- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)
- conda env create -f admet.yml
- conda activate ADMETNet
- Make install
- source ADMETNetRC
- solnet -h
- Make test
```

**Test**
```
Make test will generate example/tes_predictions.csv. Compare the predicted values 
in this file with the ones in example/tes_GCN.csv, they must be identical. Otherwise,
ADMETNet is not installed properly.
```

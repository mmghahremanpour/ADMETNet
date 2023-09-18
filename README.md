# ADMETNet
A Neural Network Framework for AMDET Prediction. 

**Installation**
```
- Install Anaconda (https://docs.anaconda.com/free/anaconda/install/index.html)
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

**Notes**
```
- Making the conda env  may take some time depending on your system.
- Executing source ADMETNetRC will define the ADMETNET env variable in your shell temporally. 
```

**Execution**
```
Run solnet -h for help! but the code can be run as 

$ solnet -if input_file.csv -of output_file.csv -nt GCN -dbp pretrained/logS/GCN/HyperParameters-Database.json

$ solnet -is "c1ccccc1" -of output_file.csv -nt GAT -dbp pretrained/logS/GAT/HyperParameters-Database.json

Please take a look at example/test.csv for formatting your input_file.csv.

```

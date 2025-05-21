# data_driven_shm

## About the project
This is the github repo that contains the code for the paper *A Data-Lean Machine Learning Approach for Damage Extent Estimation and Classification in Composite Structures Under Multiple Failure Modes*.
This repository contains the code that was used to conduct the experiments that are mentioned in the paper.

## Abstract

## Necessary Python libraries
The codes run with libraries for data processing,plotting and Machine Learning. To install them type 
```bash
pip install -r requirements.txt
```
These libraries are:

**pandas** 2.2.3

**numpy** 2.1.3

**tensorflow** 2.19.0

**scikit-learn** 1.5.0

**matplotlib** 3.10.1

**scipy** 1.15.2

**pywt** 1.8.0

**xgboost** 3.0.0

## How to run 
To run an experiment use experiment_run.py.
The script contains parameters that can be configured depending on the experiment.
These parameters are:

|                  |     Regression    |                      Classification                        | 
| ---------------- | :-------------------: | :----------------------------------------------------------: |
| Algorithm | Dummy regressor, Random forest, Linear regression, Multi Layer Perceptron  | Dummy classifier, Support Vector Machines, Random Forest, Multi Layer Perceptron| 
| Scoring method          | Mean absolute percentage error, P-value | Accuracy, F1 Macro|


## Data

The data used in the experiments can be found in the link :
(link)
## License

This project is licensed under the Apache 2 license. See `LICENSE` for details.



## Contact

If you want to contact you can send an email to the author at jimjasonp@gmail.com.


## Contributors

 <a href= "https://github.com/jimjasonp">Dimitrios Iason Papadopoulos </a> <br />


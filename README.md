# data_driven_shm

## About the project
This is the github repo that contains the code for the paper *A Data-Lean Machine Learning Approach for Damage Extent Estimation and Classification in Composite Structures Under Multiple Failure Modes*.
This repository contains the code that was used to conduct the experiments that are mentioned in the paper.

## Abstract
This study investigates the integration of Structural Health Monitoring (SHM) systems, employing active piezoelectric sensors, with Machine Learning (ML) algorithms to develop robust, AI-driven diagnostic tools for composite structures. A high-fidelity Time Domain Spectral Finite Element (TDSFE) model is developed, incorporating physically modeled piezoelectric actuators/sensors and mixed-order Layerwise Mechanics (both linear and nonlinear) to accurately simulate the structural behavior and multiple failure mechanisms—namely fiber breakage, matrix cracking, and delamination—within composite laminates. A wide range of representative damage scenarios is designed, combining both intra-laminar and inter-laminar failures distributed through the laminate thickness. These are simulated using the pitch-and-catch technique, where a Gaussian pulse is actuated and responses are collected from three spatially distributed sensors. A portion of the resulting data is used to train various ML models aimed at both classifying the type of damage and estimating its severity. This includes an extended investigation into damage classification under multiple, interacting failure modes, enhancing the diagnostic resolution of the system. The study further explores different data representation strategies for ML input and evaluates the predictive performance of the models via cross-validation. Initial results demonstrate strong potential for accurate, simulation-driven diagnostics of complex damage states in advanced composite materials.
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

**joblib** 1.4.2

**scikeras** 0.13.0

## How to run 
To run an experiment use `experiment_run.py` .
The script contains parameters that can be configured depending on the experiment.
These algorithms used in the experiments are:

|                  |     Regression    |                      Classification                        | 
| ---------------- | :-------------------: | :----------------------------------------------------------: |
| Algorithm | Dummy regressor, Random forest, Linear regression, Multi Layer Perceptron, Convolutional Neural Newtork, Long Short-Term Memory  | Dummy classifier, Support Vector Machines, Random Forest, Multi Layer Perceptron,Convolutional Neural Newtork, Long Short-Term Memory| 
| Scoring method          | Mean absolute percentage error, P-value | Accuracy, F1 Macro|


## Data

The data used in the experiments can be found in the link :

https://drive.google.com/drive/folders/1P8JL1z4u-0a4miezW2gMZdt72qGHkJFO?usp=sharing
## License

This project is licensed under the Apache 2 license. See `LICENSE` for details.



## Contact

If you want to contact you can send an email to the author at jimjasonp@gmail.com.


## Contributors
 <a href= "https://github.com/Rekchris">Christoforos Rekatsinas </a> <br />
 <a href= "https://github.com/ggianna">George Giannakopoulos </a> <br />
 <a href= "https://github.com/jimjasonp">Dimitrios Iason Papadopoulos </a> <br />


# Ensemble-Learning
## Dependencies
* python2.7
* scipy
* sklearn
## Usage
The four sub-directories contains respectively implementation for the four algorithms.  
For each algorithm, run:  
`cd $(algorithm)`  
`python $(algorithm).py`  
in which, $(algorithm) can be adaboost-svm, adaboost-dtree, bagging-svm or bagging-dtree.
The four python files have similar content, and thus similar usage.   
Method `learn()`, `reuse()` and `classify()` are provided. Call `learn()` to train new models or reuse() to
load existing trained models, and use the return values of these two methods as parameters to call 
`classify()` to generate result file.   
Generated model files are in directory `model`, while generated result files are in directory `result`.

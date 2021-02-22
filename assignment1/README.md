#CS_7641 - Assignment 1

GitHub URL: https://github.com/allalvv/CS_7641_ML_2021/edit/master/assignment1

Datasets: 
1.	“COVID-19 - Clinical Data to assess diagnosis” repository: https://www.kaggle.com/S%C3%ADrio-Libanes/covid19
2.   “Credit Card Approval”,UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/credit+approval

Models: 
1. Decision Tree
2. Neural Network
3. Boosting
4. SVM (sigmoid, rbf)
5. KNN
 
## Install project dependencies:
python 3.6

Install requirements: pip install -r requirements.txt

## Run project:

Run all experiments: 

```
python main.py --all
```

Run specific experiment: 
possiable classifyer name : dt, ada, knn, svm, nn
```
python main.py -e <classifyer name>
```
Run best results for all the models: 

```
python main.py --best
```

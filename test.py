from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold

iris = load_iris()

logreg = LogisticRegression()
loo = LeaveOneOut()


print(len(iris.data))
# scores_loo = cross_val_score(logreg, iris.data, iris.target, cv=loo)

# print(scores_loo.mean())

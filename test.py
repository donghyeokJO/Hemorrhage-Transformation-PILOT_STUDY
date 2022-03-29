from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold


# dataset

iris = load_iris()

# create object

logreg = LogisticRegression()  # model

loo = LeaveOneOut()  # LeaveOneOut model

# test validation
# LOOCV

scores_loo = cross_val_score(logreg, iris.data, iris.target, cv=loo)

# cv result

print("iris.data.shape \n{}".format(iris.data.shape))

print("cv number of partition \n{}".format(len(scores_loo)))

print("mean score_loocv \n{:.3f}".format(scores_loo.mean()))

print(scores_loo)

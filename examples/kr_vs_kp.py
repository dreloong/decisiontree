import sys
sys.path.append('../lib')

import random
import numpy as np
import decision_tree

from scipy import stats
from sklearn import cross_validation

# http://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
fname = 'data/kr_vs_kp.csv'

examples = np.genfromtxt(fname, dtype=None, delimiter=',')
random.shuffle(examples)

n_attributes = 36
attributes = []
for index in range(n_attributes):
    name = 'Attr: ' + str(index + 1)
    attributes.append(decision_tree.Attribute(name, 'discrete', index))

dtree = decision_tree.DecisionTree()

accuracies = []
kfolds = cross_validation.KFold(len(examples), n_folds=10, shuffle=True)
for train_indices, test_indices in kfolds:
    train_examples = examples[train_indices]
    dtree.fit(train_examples, attributes, pruning=True)

    test_examples = examples[test_indices]
    count = 0.0
    for example in test_examples:
        if dtree.predict(example) == example[-1]:
            count += 1
    accuracy = count / len(test_examples)
    accuracies.append(accuracy)
    print "accuracy:", accuracy

mean = np.mean(accuracies)
std = np.std(accuracies)
ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(accuracies)))
print 'Mean: {}\nStandard Deviation: {}\n95% CI: {}'.format(mean, std, ci)

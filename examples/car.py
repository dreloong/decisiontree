import sys
sys.path.append('../lib')

import numpy as np
import decision_tree

from sklearn import cross_validation

# http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
fname = 'data/car.csv'

examples = np.genfromtxt(fname, dtype=None, delimiter=',')
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

dtree = decision_tree.DecisionTree()

kfolds = cross_validation.KFold(len(examples), n_folds=10, shuffle=True)
for train_indices, test_indices in kfolds:
    train_examples = examples[train_indices]
    dtree.fit(train_examples, attributes)

    test_examples = examples[test_indices]
    count = 0.0
    for example in test_examples:
        if dtree.predict(example) == example[-1]:
            count += 1
    print "accuracy:", count / len(test_examples)

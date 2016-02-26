import sys
sys.path.append('../lib')

import random
import numpy as np
import decision_tree

from scipy import stats
from sklearn import cross_validation

# http://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
fname = 'data/cmc.csv'

examples = np.genfromtxt(fname, dtype=None, delimiter=',')
random.shuffle(examples)

attributes = [
    decision_tree.Attribute("wife's age", 'continuous', 0),
    decision_tree.Attribute("wife's education", 'discrete', 1),
    decision_tree.Attribute("husband's education", 'discrete', 2),
    decision_tree.Attribute("number of children", 'continuous', 3),
    decision_tree.Attribute("wife's religion", 'discrete', 4),
    decision_tree.Attribute("wife is working now", 'discrete', 5),
    decision_tree.Attribute("husband's occupation", 'discrete', 6),
    decision_tree.Attribute("standard-of-living index", 'discrete', 7),
    decision_tree.Attribute("media exposure", 'discrete', 8)
]

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

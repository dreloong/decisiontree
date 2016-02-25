import sys
sys.path.append('../lib')

import decision_tree

# Example 1

attributes = [
    decision_tree.Attribute('hunger', 'continuous', 0),
    decision_tree.Attribute('color', 'discrete', 1)
]

train_examples = [
    [8, 'red', 'angry'],
    [6, 'red', 'angry'],
    [7, 'red', 'angry'],
    [7, 'blue', 'not angry'],
    [2, 'red', 'not angry'],
    [3, 'blue', 'not angry'],
    [2, 'blue', 'not angry'],
    [1, 'red', 'not angry']
]

test_example = [7, 'red', 'angry']

dtree = decision_tree.DecisionTree()
dtree.fit(train_examples, attributes)
dtree.display_tree()
print dtree.predict(test_example)

# Example 2

attributes = [decision_tree.Attribute('temperature', 'continuous', 0)]

train_examples = [
    [36.6, 'healthy'],
    [37.1, 'sick'],
    [38.0, 'sick'],
    [36.7, 'healthy'],
    [39.5, 'sick'],
    [39.2, 'sick'],
    [35.7, 'sick'],
    [36.3, 'healthy'],
    [35.5, 'sick']
]

test_example = [36.5, 'healthy']

dtree = decision_tree.DecisionTree()
dtree.fit(train_examples, attributes)
dtree.display_tree()
print dtree.predict(test_example)

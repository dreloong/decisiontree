import random
import util

from Queue import Queue

class DecisionTree:

    def __init__(self, debugging=False):
        self.root = None
        self.debugging = debugging

    def fit(self, examples, attributes, fitness_metric='information_gain',
            pruning=False):
        self.attributes = attributes

        if fitness_metric == 'information_gain':
            self.fitness = util.info_gain
        else:
            raise ValueError('invalid fitness metric')

        if pruning:
            random.shuffle(examples)
            validation_set = examples[:len(examples)/3]
            training_set = examples[len(examples)/3:]
            self.root = self.build_tree(training_set)
            self.prune(validation_set)
        else:
            self.root = self.build_tree(examples)

    def predict(self, example):
        return self.descend(example, self.root)

    def build_tree(self, examples, target_index=-1):
        if len(set([example[-1] for example in examples])) == 1:
            return examples[0][-1]

        target_counts = {}
        for example in examples:
            target = example[target_index]
            if target in target_counts:
                target_counts[target] += 1
            else:
                target_counts[target] = 1

        fitness_dict = {}
        for attribute in self.attributes:
            fitness_dict[attribute] = self.fitness(examples, attribute)

        best_attr = max(fitness_dict.keys(), key=lambda k: fitness_dict[k][0])
        worst_attr = min(fitness_dict.keys(), key=lambda k: fitness_dict[k][0])
        if fitness_dict[best_attr][0] == fitness_dict[worst_attr][0]:
            best_attr = random.choice(self.attributes)

        node = TreeNode(best_attr, threshold=fitness_dict[best_attr][1])
        node.target_counts = target_counts
        subsets = {}

        if best_attr.type_ == 'discrete':
            for example in examples:
                key = example[best_attr.index]
                if key in subsets:
                    subsets[key].append(example)
                else:
                    subsets[key] = [example]
        else:
            subsets[' >= '] = []
            subsets[' < '] = []
            for example in examples:
                if example[best_attr.index] >= node.threshold:
                    subsets[' >= '].append(example)
                else:
                    subsets[' < '].append(example)

        # examples with identical attributes but different classes
        if len(subsets) == 1:
            return examples[0][-1]

        for value, subset in subsets.iteritems():
            node.children[value] = self.build_tree(subset)

        return node

    def descend(self, example, node):
        if not isinstance(node, TreeNode):
            return node

        value = example[node.attribute.index]

        if node.attribute.type_ == 'continuous':
            if value >= node.threshold:
                return self.descend(example, node.children[' >= '])
            else:
                return self.descend(example, node.children[' < '])

        # discrete attribute
        if value not in node.children:
            return None
        return self.descend(example, node.children[value])

    def prune(self, examples, target_index=-1):
        init_accuracy = self.accuracy(examples, target_index)
        max_accuracy = init_accuracy
        q = Queue()
        q.put(self.root)
        nodes = [self.root]

        while not q.empty():
            node = q.get()
            for child in node.children.values():
                if isinstance(child, TreeNode):
                    q.put(child)
                    nodes.append(child)

        while len(nodes) > 0:
            node = nodes.pop()
            for value in node.children:
                child = node.children[value]
                if not isinstance(child, TreeNode):
                    continue

                majority_target = max(
                    child.target_counts.keys(),
                    key=lambda x: child.target_counts[x]
                )
                node.children[value] = majority_target
                current_accuracy = self.accuracy(examples, target_index)
                if current_accuracy > max_accuracy:
                    max_accuracy = current_accuracy
                else:
                    node.children[value] = child

        if self.debugging:
            print 'Accuracy improved from {} to {} after pruning.'.format(
                init_accuracy,
                max_accuracy)

    def accuracy(self, examples, target_index=-1):
        count = 0
        for example in examples:
            if example[target_index] == self.predict(example):
                count += 1
        return 1.0 * count / len(examples)

    def display_tree(self):
        self.display_tree_dfs(self.root, 0)

    def display_tree_dfs(self, node, level):
        if not isinstance(node, TreeNode):
            return

        for value, child in node.children.iteritems():
            target_info = child
            if isinstance(child, TreeNode):
                target_info = child.target_counts

            if node.attribute.type_ == 'continuous':
                print '{}{}{}{} [{}]'.format(
                    '  ' * level,
                    node.attribute.name,
                    value,
                    node.threshold,
                    target_info)
            else:
                print '{}{} = {} [{}]'.format(
                    '  ' * level,
                    node.attribute.name,
                    value,
                    target_info)
            self.display_tree_dfs(child, level + 1)

class TreeNode:

    def __init__(self, attribute, threshold=None):
        self.attribute = attribute
        self.threshold = threshold
        self.children = {}
        self.target_counts = {}

class Attribute:

    def __init__(self, name, type_, index):
        self.name = name
        self.type_ = type_
        self.index = index

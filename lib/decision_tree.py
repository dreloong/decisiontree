import random
import util

class DecisionTree:

    def __init__(self):
        self.root = None

    def fit(self, examples, attributes, fitness_metric='information_gain'):
        self.attributes = attributes

        if fitness_metric == 'information_gain':
            self.fitness = util.info_gain
        else:
            raise ValueError('invalid fitness metric')

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

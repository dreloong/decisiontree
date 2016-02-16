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

    def build_tree(self, examples):
        if len(set([example[-1] for example in examples])) == 1:
            return examples[0][-1]

        fitness_dict = {}
        for index, attribute in enumerate(self.attributes):
            fitness_dict[attribute] = self.fitness(examples, index)

        best_attr = max(fitness_dict.keys(), key=lambda k: fitness_dict[k])
        worst_attr = min(fitness_dict.keys(), key=lambda k: fitness_dict[k])
        if fitness_dict[best_attr] == fitness_dict[worst_attr]:
            best_attr = random.choice(self.attributes)

        node = TreeNode(best_attr)

        subsets = {}
        index = self.attributes.index(best_attr)
        for example in examples:
            key = example[index]
            if key in subsets:
                subsets[key].append(example)
            else:
                subsets[key] = [example]

        # examples with identical attributes but different classes
        if len(subsets) == 1:
            return examples[0][-1]

        for value, subset in subsets.iteritems():
            node.children[value] = self.build_tree(subset)

        return node

    def descend(self, example, node):
        if not isinstance(node, TreeNode):
            return node

        index = self.attributes.index(node.attribute)
        if example[index] not in node.children:
            return None
        return self.descend(example, node.children[example[index]])

    def display_tree(self):
        self.display_tree_dfs(self.root, 0)

    def display_tree_dfs(self, node, level):
        if not isinstance(node, TreeNode):
            print '  ' * level + node
            return

        for value, child in node.children.iteritems():
            print '  ' * level + node.attribute + ' = ' + value
            self.display_tree_dfs(child, level + 1)

class TreeNode:

    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

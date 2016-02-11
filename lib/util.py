import math

def info_gain(examples, attribute_index, target_index=-1):
    subsets_dict = {}
    for example in examples:
        key = example[attribute_index]
        if key in subsets_dict:
            subsets_dict[key].append(example)
        else:
            subsets_dict[key] = [example]

    gain = entropy(examples, target_index)
    for subset in subsets_dict.values():
        proportion = 1.0 * len(subset) / len(examples)
        gain -= proportion * entropy(subset)
    return gain

def entropy(examples, target_index=-1):
    target_counts = {}
    for example in examples:
        target = example[target_index]
        if target in target_counts:
            target_counts[target] += 1
        else:
            target_counts[target] = 1

    result = 0
    for count in target_counts.values():
        proportion = 1.0 * count / len(examples)
        result -= proportion * math.log(proportion, 2)
    return result

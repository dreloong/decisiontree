import math
import numpy as np

def info_gain(examples, attribute, target_index=-1):
    if attribute.type_ == 'discrete':
        return info_gain_discrete(examples, attribute.index, target_index)
    if attribute.type_ == 'continuous':
        return info_gain_continuous(examples, attribute.index, target_index)
    raise ValueError('invalid attibute type')

def info_gain_discrete(examples, attribute_index, target_index=-1):
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
    return gain, None

def info_gain_continuous(examples, attribute_index, target_index=-1):
    values = sorted(set([example[attribute_index] for example in examples]))
    if len(values) == 1:
        return -1, 0

    thresholds = []
    for i in range(1, len(values)):
        thresholds.append(0.5 * (values[i-1] + values[i]))

    gains = []
    for threshold in thresholds:
        subsets = [[], []]
        for example in examples:
            if example[attribute_index] >= threshold:
                subsets[0].append(example)
            else:
                subsets[1].append(example)

        gain = entropy(examples, target_index)
        for subset in subsets:
            proportion = 1.0 * len(subset) / len(examples)
            gain -= proportion * entropy(subset)

        gains.append(gain)

    index = np.argmax(gains)
    return gains[index], thresholds[index]

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

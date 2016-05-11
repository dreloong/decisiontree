# Decision Tree
Simple implementation of decision tree.

## Usage

```python
import decision_tree
dtree = decision_tree.DecisionTree()

# Training
dtree.fit(train_examples, attributes, pruning=True)
# Display
dtree.display_tree()
# Prediction
dtree.predict(test_example)
```

## References

Tutorials:
- [Decision Trees and Political Party Classification](https://jeremykun.com/tag/decision-trees/)
- [Decision Tree Learning in Ruby](https://www.igvita.com/2007/04/16/decision-tree-learning-in-ruby/)

Dataset:
- [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/)

## License

    Copyright [2016] [Xiaofei Long]
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

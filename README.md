# scene_text_framework
A lightweight training/evaluation framework for scene text tasks.


## Quick Start!

1. Clone target model source file into dir `models`, there is already a demo of `SVTR`, such as `models/my-model/my-model.py`
2. Create an `adapter.py` in your model dir and instantiate it by inhereting the base class `ModelAdapter`, such as `models/my-model/adapter.py`
3. If `CTCLoss` can not satisfy the training object, add new criterions in `arch/criterion.py`
4. If `Accuracy` can not satisfy the evaluating phase, add new metrics in `arch/validator.py`
5. We have provided the demo of `SVTR`, thanks for the repo [SVTR Pytoch](https://github.com/j-river/svtr-pytorch)
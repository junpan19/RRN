![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch 1.1](https://img.shields.io/badge/pytorch-1.1-yellow.svg)

# Recurrent Residual Network for Video Super-resolution (RRN)

The *official* implementation for the [Revisiting Temporal Modeling for Video
Super-resolution] which is accepted by [BMVC-2020].

![framework](figs/framework.PNG)

### Train
We utilize 4 GTX-1080TI GPUs for training.
```
python main.py
```

### Test
We utilize 1 GTX-1080TI GPU for testing.
Test the trained model with best performance by
```
python test.py
```

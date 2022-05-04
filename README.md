# DreamBC: Dreaming with Bisimulation for Control

Here ```alg = dreamer,dreambc```

Training on ```walker-walk```: ```python alg_main.py --env walker-walk```

Task Generalization on ```walker-stand```: ```python alg_main.py --env walker-stand --model results/walker-walk/alg/models.pth```

Feature Generalization on ```walker-walk```: ```python alg_main.py --env walker-walk --img-source color --id alg-color```

Render Episodes: ```python alg_main.py --env walker-walk --model results/walker-walk/alg/models.pth --agent-model results/walker-walk/alg/models.pth --test --test-episodes 5 --video```

# Citations
PlaNet: [Paper](https://arxiv.org/abs/1811.04551) [Code](https://github.com/google-research/planet) [PyTorch](https://github.com/Kaixhin/PlaNet)

Dreamer: [Paper](https://arxiv.org/abs/1912.01603) [Code](https://github.com/danijar/dreamer) [PyTorch](https://github.com/juliusfrost/dreamer-pytorch)

DBC: [Paper](https://arxiv.org/abs/2006.10742) [Code](https://github.com/facebookresearch/deep_bisim4control)

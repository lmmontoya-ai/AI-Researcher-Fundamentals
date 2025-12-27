# AI Researcher Fundamentals

A learning roadmap to build research-grade intuition by exploring, explaining, and coding each topic.

## Learning Loop

- **Explore**: intuition, visuals, and toy data to build a mental model
- **Explain**: concise write-up and derivations in the notebook
- **Code**: from-scratch implementation plus a library baseline

---

## Roadmap

---

### 1) Linear Models and Evaluation *(in progress)*
The foundation—understanding linear models deeply unlocks everything else.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Linear Regression | Closed-form solution, gradient descent, geometric interpretation ✅ |
| 2 | Polynomial Features & Overfitting | Feature engineering, capacity, visualizing overfitting |
| 3 | Loss Functions & Metrics | MSE, MAE, Huber, R², adjusted R² |
| 4 | Train/Validation/Test Splits | Data splitting strategies, information leakage |
| 5 | Feature Scaling | Standardization, normalization, when and why |
| 6 | Bias-Variance Tradeoff | Decomposition, underfitting vs overfitting |
| 7 | Regularization (Ridge & Lasso) | L1 vs L2 geometry, sparsity, ElasticNet |
| 8 | Bayesian Linear Regression | Prior over weights, posterior predictive, uncertainty |

---

### 2) Core Supervised ML
Classical ML algorithms every researcher should understand deeply.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Logistic Regression | Sigmoid, cross-entropy, decision boundary, multiclass |
| 2 | k-Nearest Neighbors | Distance metrics, curse of dimensionality, KD-trees |
| 3 | Naive Bayes | Conditional independence, text classification, Laplace smoothing |
| 4 | Decision Trees | Information gain, Gini impurity, pruning, interpretability |
| 5 | Random Forests | Bagging, feature importance, out-of-bag error |
| 6 | Gradient Boosting | AdaBoost, XGBoost, LightGBM, bias-variance in ensembles |
| 7 | Support Vector Machines | Margins, kernels, dual formulation, kernel trick |
| 8 | Model Selection & Cross-Validation | k-fold, stratified, nested CV, hyperparameter tuning |

---

### 3) Unsupervised Learning and Representation
Learning structure without labels.

| # | Topic | Description |
|---|-------|-------------|
| 1 | K-Means Clustering | Lloyd's algorithm, initialization (k-means++), elbow method |
| 2 | Hierarchical Clustering | Agglomerative, dendrograms, linkage criteria |
| 3 | DBSCAN | Density-based clustering, core points, noise handling |
| 4 | Gaussian Mixture Models | Soft clustering, EM algorithm derivation |
| 5 | Principal Component Analysis | Covariance, eigenvectors, variance explained, reconstruction |
| 6 | Kernel PCA | Nonlinear dimensionality reduction, kernel trick |
| 7 | t-SNE | Perplexity, crowding problem, visualization best practices |
| 8 | UMAP | Manifold learning, topological data analysis intuition |

---

### 4) Optimization and Probabilistic ML
The engine behind learning and principled uncertainty.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Gradient Descent Variants | Batch, stochastic, mini-batch, convergence analysis |
| 2 | Momentum & Adaptive Methods | Momentum, Nesterov, AdaGrad, RMSprop, Adam |
| 3 | Second-Order Methods | Newton's method, BFGS, L-BFGS, Hessian approximation |
| 4 | Constrained Optimization | Lagrange multipliers, KKT conditions, duality |
| 5 | Convex vs Non-Convex | Convexity, local minima, saddle points, loss landscapes |
| 6 | Maximum Likelihood Estimation | Likelihood function, log-likelihood, MLE derivations |
| 7 | MAP & Bayesian Inference | Priors, posteriors, conjugate priors, MCMC basics |
| 8 | Calibration & Uncertainty | Reliability diagrams, temperature scaling, epistemic vs aleatoric |

---

### 5) Neural Network Fundamentals
Building blocks of deep learning.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Perceptron & MLPs | Linear classifiers, universal approximation, architecture |
| 2 | Backpropagation Derivation | Chain rule, computational graphs, gradient flow |
| 3 | Activation Functions | ReLU, Leaky ReLU, GELU, Swish, dying ReLU problem |
| 4 | Weight Initialization | Xavier/Glorot, He, variance preservation |
| 5 | Batch Normalization | Internal covariate shift, training vs inference |
| 6 | Layer & Group Normalization | Batch-independent normalization, when to use each |
| 7 | Dropout & Regularization | Co-adaptation, weight decay, early stopping |
| 8 | Optimization Tricks | Learning rate schedules, warmup, gradient clipping |

---

### 6) Convolutional Networks
Vision and spatial understanding.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Convolution Operations | Filters, stride, padding, receptive field |
| 2 | Pooling & Downsampling | Max pooling, average pooling, strided convolutions |
| 3 | Translational Equivariance | Why CNNs work, inductive biases |
| 4 | Classic Architectures | LeNet, AlexNet, VGG, architecture evolution |
| 5 | Residual Networks | Skip connections, gradient highways, ResNet variations |
| 6 | Modern CNNs | EfficientNet, ConvNeXt, RegNet |
| 7 | Vision Transformers | Patch embeddings, ViT, hybrid architectures |
| 8 | Transfer Learning | Pretrained features, fine-tuning strategies |

---

### 7) Sequence Models and Attention
Processing sequential data and the attention revolution.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Recurrent Neural Networks | Unrolling, hidden states, sequence modeling |
| 2 | LSTM & GRU | Gates, memory cells, vanishing gradient solution |
| 3 | Sequence-to-Sequence | Encoder-decoder, teacher forcing |
| 4 | Attention Mechanisms | Bahdanau vs Luong attention, alignment |
| 5 | Transformer Architecture | Self-attention, multi-head attention, feedforward |
| 6 | Positional Encodings | Sinusoidal, learned, RoPE, ALiBi |
| 7 | Attention Variants | Flash attention, linear attention, sparse attention |
| 8 | Embeddings & Representations | Word2Vec, GloVe, contextualized embeddings |

---

### 8) Large Language Models
Modern foundation models and their techniques.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Scaling Laws | Chinchilla, compute-optimal training, emergent abilities |
| 2 | Pretraining Objectives | Masked LM, causal LM, T5-style span corruption |
| 3 | Tokenization | BPE, WordPiece, SentencePiece, vocabulary size |
| 4 | Architecture Deep Dive | GPT, BERT, T5, LLaMA architecture choices |
| 5 | Prompting & In-Context Learning | Zero-shot, few-shot, chain-of-thought |
| 6 | Fine-Tuning Techniques | Full fine-tuning, LoRA, adapters, PEFT |
| 7 | RLHF & Alignment | Reward modeling, PPO for LLMs, DPO |
| 8 | Multimodal Models | CLIP, vision-language models, architecture patterns |

---

### 9) Generative Modeling
Creating new data from learned distributions.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Autoencoders | Bottleneck, reconstruction, latent space |
| 2 | Variational Autoencoders | ELBO derivation, reparameterization trick, KL term |
| 3 | GAN Fundamentals | Minimax game, generator/discriminator, mode collapse |
| 4 | GAN Variants & Training | DCGAN, StyleGAN, training stability tricks |
| 5 | Normalizing Flows | Invertible transforms, Jacobian determinant, RealNVP |
| 6 | Score-Based Models | Score matching, Langevin dynamics |
| 7 | Diffusion Models | Forward/reverse process, DDPM, noise schedules |
| 8 | Evaluation Metrics | FID, IS, precision/recall, perceptual metrics |

---

### 10) Reinforcement Learning
Learning from interaction with environments.

| # | Topic | Description |
|---|-------|-------------|
| 1 | MDPs & Bellman Equations | States, actions, rewards, value functions |
| 2 | Dynamic Programming | Policy evaluation, policy iteration, value iteration |
| 3 | Monte Carlo Methods | Episode sampling, first-visit vs every-visit |
| 4 | Temporal Difference Learning | TD(0), TD(λ), eligibility traces |
| 5 | Q-Learning & DQN | Off-policy learning, experience replay, target networks |
| 6 | Policy Gradient Methods | REINFORCE, baseline subtraction, variance reduction |
| 7 | Actor-Critic & PPO | A2C, A3C, PPO clipping, GAE |
| 8 | Practical RL | Reward shaping, exploration strategies, debugging RL |

---

### 11) Interpretability and Safety
Understanding and trusting models.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Feature Attribution | Saliency maps, gradient × input, SmoothGrad |
| 2 | Integrated Gradients & SHAP | Path methods, Shapley values, TreeSHAP |
| 3 | Attention Visualization | Attention rollout, attention flow, limitations |
| 4 | Probing Representations | Linear probes, diagnostic classifiers |
| 5 | Mechanistic Interpretability | Circuits, features, superposition |
| 6 | Adversarial Robustness | FGSM, PGD, adversarial training |
| 7 | AI Safety Fundamentals | Alignment problem, reward hacking, distributional shift |
| 8 | Responsible AI | Fairness metrics, bias mitigation, model cards |

---

### 12) Learning Theory
Why and when does learning work?

| # | Topic | Description |
|---|-------|-------------|
| 1 | Bias-Variance Deep Dive | Decomposition proof, model complexity curves |
| 2 | PAC Learning | Probably approximately correct, sample complexity |
| 3 | VC Dimension | Shattering, growth function, VC bounds |
| 4 | Rademacher Complexity | Data-dependent complexity, generalization bounds |
| 5 | Regularization Theory | Implicit regularization, flat minima |
| 6 | Double Descent | Modern interpolation regime, benign overfitting |
| 7 | Neural Tangent Kernel | Infinite-width limit, lazy training |
| 8 | Lottery Ticket Hypothesis | Sparse networks, pruning at initialization |

---

### 13) Research Practice
Skills for doing ML research.

| # | Topic | Description |
|---|-------|-------------|
| 1 | Paper Reading Strategies | 3-pass method, critical reading, note-taking |
| 2 | Literature Review | Finding papers, citation graphs, staying current |
| 3 | Reproducing Papers | Common pitfalls, missing details, hyperparameters |
| 4 | Experiment Design | Controls, ablations, statistical significance |
| 5 | Experiment Tracking | Logging, wandb/mlflow, version control for ML |
| 6 | Statistical Testing | Confidence intervals, hypothesis testing, multiple comparisons |
| 7 | Scientific Writing | Paper structure, figures, clear communication |
| 8 | Reproducibility & Open Science | Code release, data documentation, model cards |

---

## Project Structure

```
.
├── LICENSE
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── src/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_mathematical_foundations/
│   ├── 01_linear_models/
│   │   └── 1_Linear_Regression.ipynb  ✅
│   ├── 02_core_supervised/
│   ├── 03_unsupervised_representation/
│   ├── 04_optimization_probabilistic/
│   ├── 05_neural_network_fundamentals/
│   ├── 06_convolutional_networks/
│   ├── 07_sequence_attention/
│   ├── 08_large_language_models/
│   ├── 09_generative_modeling/
│   ├── 10_reinforcement_learning/
│   ├── 11_interpretability_safety/
│   ├── 12_learning_theory/
│   └── 13_research_practice/
└── notes/
    └── ... (mirrors notebooks structure)
```

---

## Resources

### Textbooks
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Pattern Recognition and Machine Learning* — Bishop
- *Deep Learning* — Goodfellow, Bengio, Courville
- *Reinforcement Learning: An Introduction* — Sutton & Barto
- *Understanding Deep Learning* — Prince

### Courses
- Stanford CS229 (Machine Learning)
- Stanford CS231n (CNNs for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- UC Berkeley CS285 (Deep Reinforcement Learning)
- MIT 6.S898 (Deep Learning)

### Papers to Start
- "Attention Is All You Need" — Vaswani et al.
- "Deep Residual Learning" — He et al.
- "Adam: A Method for Stochastic Optimization" — Kingma & Ba
- "Generative Adversarial Networks" — Goodfellow et al.
- "BERT: Pre-training of Deep Bidirectional Transformers" — Devlin et al.

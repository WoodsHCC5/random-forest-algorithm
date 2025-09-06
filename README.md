# Random Forest Algorithm (vs. Single Decision Tree)

## Why Random Forest When We Already Have Decision Trees?

A single decision tree is:
- Easy to interpret.
- Fast to train.
- Able to model non‑linear relationships.

But it is also:
- High variance: small changes in training data can drastically change the tree.
- Prone to overfitting (especially deep trees).
- Greedy: each split is made locally; it cannot revise earlier choices.

A Random Forest reduces variance by averaging many de-correlated trees. Each tree overfits differently; their aggregation (majority vote or average) cancels out idiosyncratic errors while preserving true signal.

## Similarities and Differences

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| Base learner | Single tree | Many trees (ensemble) |
| Training data | Full dataset | Bootstrap sample per tree (sampling with replacement) |
| Feature use at split | All features considered | Random subset of features at each split (feature subsampling) |
| Variance | High | Much lower (averaging) |
| Bias | Low–moderate | Slightly higher than a single fully grown tree (due to randomness) |
| Interpretability | High | Low for entire forest (individual trees still interpretable) |
| Overfitting tendency | High if grown deep | Greatly reduced |
| Prediction (classification) | Leaf majority class | Majority vote across trees |
| Prediction (regression) | Leaf mean target | Mean of tree predictions |

## How a Decision Tree Is Trained (Classification)

1. Start with all training samples at the root.
2. For each candidate feature (and threshold):
   - Split the data into left/right (or children for categorical).
   - Compute impurity (e.g., entropy or Gini) before and after the split.
   - Choose the split that maximizes information gain (impurity reduction).
3. Recurse on each child until:
   - Pure node (all same class), or
   - Max depth / min samples stopping condition.
4. Assign a class (majority) to each leaf.

### Entropy (Classification Impurity)
Entropy measures label disorder:
Entropy(S) = - Σ p_i * log2(p_i)
Where p_i = proportion of class i in node S.
- Entropy = 0 when all samples are same class (pure).
- Higher entropy = more mixed labels.

(Alternative: Gini impurity = 1 - Σ p_i²)

## How a Random Forest Is Trained

For each of T trees:
1. Bootstrap: Sample N instances with replacement from original dataset of size N (some omitted, some duplicated). (Bagging = Bootstrap + Aggregating.)
2. Grow a full (often unpruned) decision tree:
   - At each split, randomly select a subset of m features out of total p.
   - Only these m are evaluated for the best split (injects de-correlation).
3. Store the trained tree.

Prediction:
- Classification: majority vote over tree outputs.
- Regression: arithmetic mean of tree numerical outputs.

### Why Bagging Helps
- Bootstrap sampling creates diverse training subsets.
- Many trees overfit differently.
- Aggregation averages away noise (variance reduction) while preserving underlying structure.

### Why Random Feature Subsets Help
If the dataset has one very strong predictor, every tree’s root split would be identical without randomness → trees become highly correlated → averaging gives little variance reduction. Restricting each split to a random subset forces variety.

## Feature Subset Size (m_try)

Common heuristics (p = total features):
- Classification: m = √p (introduced/popularized in Breiman’s Random Forests (2001)).
- Regression: m ≈ p / 3 (Breiman, 2001).
- Earlier work (Ho, 1995; 1998 “Random Decision Forests”) used variants like log2(p) + 1.

References (see bottom) attribute:
- Random feature selection concept to Tin Kam Ho.
- Canonical defaults (√p, p/3) to Leo Breiman.

No single “ideal” size—tune m via validation.

## Sensitivity to Training Data

- Decision Tree: High variance. Small perturbations → different splits early in the tree → cascading structural differences.
- Random Forest: Averaging many high-variance trees reduces variance roughly proportional to correlation ρ between trees: Var(average) ≈ ρσ² + (1 - ρ)σ² / T. Random feature selection + bootstrapping reduce ρ.

## Simple Hand-Crafted Example (Binary Classification)

Assume dataset (p = 3 features: A, B, C):

| Id | A | B | C | Class |
|----|---|---|---|-------|
| 1  | 0 | 1 | 2 | Yes   |
| 2  | 1 | 0 | 1 | No    |
| 3  | 1 | 1 | 2 | Yes   |
| 4  | 0 | 0 | 0 | No    |
| 5  | 1 | 0 | 2 | Yes   |
| 6  | 0 | 1 | 1 | No    |

Let T = 3 trees, m = √3 ≈ 1 feature per split (force extreme randomness for illustration).

### Step 1: Bootstrap Samples (example)
Tree 1 sample (Ids): [3,5,5,2,6,1]
Tree 2 sample: [4,2,2,6,1,3]
Tree 3 sample: [5,5,1,4,6,3]

(Out-of-bag (OOB) examples differ per tree; can estimate accuracy without separate validation set.)

### Step 2: Grow Each Tree
At each node:
- Randomly pick 1 feature (since m=1 here).
- Choose best threshold for that feature within the node.

Because only one feature is examined per split, trees differ markedly.

### Toy ASCII Illustration

Tree 1 (example):
A?
 ├─ A=0 → Predict No
 └─ A=1 → Predict Yes

Tree 2:
C?
 ├─ C <= 1 → Predict No
 └─ C > 1  → Predict Yes

Tree 3:
B?
 ├─ B=0 → Predict No
 └─ B=1 → (fallback majority Yes)

Prediction for a new point (A=1,B=1,C=2):
- Tree1: Yes
- Tree2: Yes (C>1)
- Tree3: Yes (B=1)
Majority → Yes.

Another sample (A=0,B=0,C=1):
- Tree1: No
- Tree2: No (C<=1)
- Tree3: No (B=0)
Result → No.

Bad decisions from individual trees may cancel out because their mistakes are not aligned.

### Mermaid Diagram (Forest Overview)

```mermaid
graph TD
  A[Decision Forest] --> |Vote/Majority| B[Prediction]
  A --> C[Tree 1]
  A --> D[Tree 2]
  A --> E[Tree 3]
  subgraph Tree1 [Decision Tree 1]
    direction TB
    C1[Feature A?] --> |Yes| C2[Class: Yes]
    C1 --> |No| C3[Feature B?]
    C3 --> |Yes| C4[Class: Yes]
    C3 --> |No| C5[Class: No]
    C1:::tree
  end
  subgraph Tree2 [Decision Tree 2]
    direction TB
    D1[Feature C?] --> |<= 1| D2[Class: No]
    D1 --> |> 1| D3[Class: Yes]
    D1:::tree
  end
  subgraph Tree3 [Decision Tree 3]
    direction TB
    E1[Feature B?] --> |0| E2[Class: No]
    E1 --> |1| E3[Class: Yes]
    E1:::tree
  end
  classDef tree fill:#f9f9f9,stroke:#333,stroke-width:2px;
  class C1,C3,C5,D1,D3,E1,E3 tree;
````````

### Bagging + Feature Subsampling
- Bootstrap introduces variation in row space.
- Random feature subset (m) introduces variation in column space.
Together they reduce tree correlation → better variance reduction.

## Overfitting vs Generalization

- If all trees used all data and all features at every split, trees would be nearly identical → averaging gives minimal benefit and the forest could collectively overfit.
- Injected randomness (bootstrap + feature subsets) encourages diverse decision boundaries, improving generalization.

## Using Random Forest for Regression

Regression problem: Predict a continuous numeric target y given feature vector x (e.g., predict house price).

Decision Tree (Regression):
- At each split choose feature/threshold minimizing variance (or MSE) of child nodes.
- Leaf prediction = mean of y values in that leaf.

Random Forest (Regression):
- Train trees on bootstraps, with random feature subsets.
- Each tree outputs a numeric prediction (leaf mean).
- Final prediction = average (sometimes median) of tree outputs.
- Reduces variance while retaining low bias.

## When to Use What

- Single decision tree: Need transparency, quick baseline, or interpretable rules.
- Random forest: Need strong accuracy out-of-the-box, robustness, handles mixed feature types, limited tuning, good default for tabular data.

## Practical Tips

- Tune number of trees (T): More trees rarely hurt; diminishing returns after variance stabilizes.
- Tune m (m_try): Start with defaults (√p classification, p/3 regression). Validate.
- Use Out-of-Bag (OOB) error: Evaluate performance without a separate validation set (aggregate predictions for samples not included in a tree’s bootstrap).
- Control overfitting: Typically trees are grown fully (no pruning); randomness + averaging is the regularizer.

## Key Terms

- Bootstrap: Sampling with replacement from original dataset to create training subsets.
- Aggregating: Combining predictions (vote or average). Bootstrap + Aggregating = Bagging.
- Feature Subset (Random Subspace): Randomly restricting features considered at each split.
- OOB Error: Error estimated using data not sampled in a tree’s bootstrap.
- Impurity: Measure of node disorder (Entropy/Gini for classification; variance/MSE for regression).

## References (Foundational Works)

1. Breiman, L. (1996). Bagging Predictors. Machine Learning, 24(2), 123–140.  
2. Ho, T.K. (1995). Random Decision Forests. Proceedings of 3rd International Conference on Document Analysis and Recognition.  
3. Ho, T.K. (1998). The Random Subspace Method for Constructing Decision Forests. IEEE TPAMI, 20(8), 832–844.  
4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.  (Introduced standard defaults: m = √p for classification; m ≈ p/3 for regression.)

(Paraphrased—consult originals for precise formal definitions.)

## Summary

Random Forest = Many high-variance trees + diversity mechanisms (bootstrap + random feature subsets) + aggregation → Lower variance, strong generalization, minimal tuning. Decision trees are the interpretable core model; random forests industrialize them into a powerful ensemble.

### How can bias be higher in a Random Forest than in a single Decision Tree?

Bias = systematic difference between the expected model prediction and the true target function. A fully grown decision tree (no pruning, all features available at every split, trained on the entire dataset) tends to have very low bias because it can carve out extremely fine partitions (often to pure leaves). A Random Forest can introduce a *slight* increase in bias relative to that single best-fit deep tree due to its injected randomness:

#### Sources of Additional Bias in Random Forests
1. Bootstrap Sampling  
   Each tree sees on average only about 63% of unique training samples (the rest are out-of-bag). With fewer effective samples per tree, some splits are less precise; individually, trees are a bit more biased than a tree trained on the full dataset. Averaging reduces variance but cannot fully remove that per-tree underfitting signal.
2. Random Feature Subsets (m_try)  
   At a node the best global split (using all features) might not be considered if the informative feature is absent from the sampled subset. The chosen split is then a second-best surrogate, slightly degrading the decision boundary. Repeating this over many levels pushes the expected tree structure toward a smoother, less sharply aligned boundary → mild bias increase.
3. Strong Dominant Predictor Scenario  
   Single tree: uses dominant feature at root → near-optimal partition.  
   Random forest with small m_try: many trees cannot use that feature first → blended weaker partitions shift boundary.
4. Depth Constraints + Randomness  
   Additional regularization (max_depth, min_samples_leaf) compounds bias. Randomness cannot recover lost structure.
5. Smoothing Effect of Averaging  
   Probability or regression surfaces become smoother (good for variance) but can underfit sharp boundaries.

#### When the Bias Increase Is Noticeable
- Very small m_try relative to p.
- Very small datasets.
- Heavy simultaneous regularization.
- Targets with sharp discontinuities or very localized interactions.

#### Why It Is Usually Acceptable
Variance reduction dominates; overall generalization error typically falls despite a small bias uptick.

#### Mitigating Excess Bias
- Increase m_try.
- Allow deeper trees / smaller leaves.
- Ensure enough trees (stabilize expectation).
- Avoid Extra Trees if bias already problematic.
- Consider boosting for extremely sharp functions.

#### Takeaway
Random Forests trade tiny bias increases for large variance reductions—net win in most real datasets.

---

## Bias vs Variance (Video)


**Thumbnail (click to open video):**  
[![Bias vs Variance Video](https://img.youtube.com/vi/tUs0fFo7ki8/hqdefault.jpg)](https://www.youtube.com/watch?v=tUs0fFo7ki8 "Bias vs Variance")
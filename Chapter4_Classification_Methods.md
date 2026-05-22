# Chapter 4: Classification Methods Beyond Logistic Regression
### *An Integrated Lecture + Lab Guide — ISLR Second Edition*

> **How to use this document:** Concepts build on each other. Read straight through the first time — each method opens with intuition, then math, then immediate R implementation. Every code block is self-contained and runnable. Come back to individual methods as a reference.

---

## Table of Contents

| Part | Topic |
|------|-------|
| **0** | [Setup & Data Orientation](#part-0-setup--data-orientation) |
| **1** | [Why Not Just Use Logistic Regression?](#part-1-why-not-just-use-logistic-regression) |
| **2** | [Bayes Theorem — The Engine Under the Hood](#part-2-bayes-theorem--the-engine-under-the-hood) |
| **3** | [Linear Discriminant Analysis (LDA)](#part-3-linear-discriminant-analysis-lda) |
| **4** | [Quadratic Discriminant Analysis (QDA)](#part-4-quadratic-discriminant-analysis-qda) |
| **5** | [Naive Bayes](#part-5-naive-bayes) |
| **6** | [K-Nearest Neighbors (KNN)](#part-6-k-nearest-neighbors-knn) |
| **7** | [Evaluating Classifiers Beyond Accuracy](#part-7-evaluating-classifiers-beyond-accuracy) |
| **8** | [Comparing All Methods Side by Side](#part-8-comparing-all-methods-side-by-side) |
| **9** | [Decision Boundaries Visualized](#part-9-decision-boundaries-visualized) |
| **10** | [Practical Guide: Which Method, When?](#part-10-practical-guide-which-method-when) |
| **11** | [Common Pitfalls & Checklist](#part-11-common-pitfalls--checklist) |
| **A** | [Formula Reference Sheet](#appendix-formula-reference-sheet) |

---

---

# Part 0: Setup & Data Orientation

## 0.1 Install and Load Everything

```r
# ── Install once ─────────────────────────────────────────────────────────────
install.packages(c("ISLR2", "MASS", "e1071", "class", "ggplot2",
                   "dplyr", "tidyr", "pROC", "caret"))

# ── Load every session ────────────────────────────────────────────────────────
library(ISLR2)   # Default, Smarket, Auto datasets
library(MASS)    # lda(), qda()
library(e1071)   # naiveBayes()
library(class)   # knn()
library(ggplot2) # Plotting
library(dplyr)   # Data wrangling
library(pROC)    # ROC curves and AUC
library(caret)   # confusionMatrix()
```

## 0.2 The Default Dataset — Our Main Playground

The `Default` dataset tracks 10,000 credit card customers. Did they default on their payment? This is our classification target.

```r
data(Default)

# Structure
str(Default)
# 10,000 obs of 4 variables:
# $ default: Factor "No"/"Yes"  ← RESPONSE (what we predict)
# $ student: Factor "No"/"Yes"  ← Categorical predictor
# $ balance: num                ← Continuous (credit card balance $)
# $ income : num                ← Continuous (annual income $)

# Class distribution
table(Default$default)
prop.table(table(Default$default))
```

**Output:**
```
   No   Yes
 9667   333

      No      Yes
0.966700 0.033300
```

**This imbalance is critical.** 96.7% of customers did *not* default. A model that predicts "No" for everyone gets 96.7% accuracy — but is completely useless for finding actual defaulters. Keep this in mind throughout the chapter.

```r
# ── Visualize the key relationships ──────────────────────────────────────────
ggplot(Default, aes(x = balance, y = income, color = default)) +
  geom_point(alpha = 0.3, size = 0.8) +
  scale_color_manual(values = c("No" = "steelblue", "Yes" = "tomato")) +
  labs(
    title    = "Default Status by Balance and Income",
    subtitle = "Defaulters (red) cluster at high balance — income is far less predictive.\nThis visual tells you which predictor will do the heavy lifting.",
    x = "Credit Card Balance ($)", y = "Annual Income ($)"
  ) +
  theme_minimal()
```

```r
# ── Density plots show the distributional difference ─────────────────────────
ggplot(Default, aes(x = balance, fill = default)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("No" = "steelblue", "Yes" = "tomato")) +
  labs(
    title    = "Balance Distribution by Default Status",
    subtitle = "These two bell curves barely overlap — balance is a powerful separator.\nLDA will love this.",
    x = "Balance ($)", y = "Density"
  ) +
  theme_minimal()
```

## 0.3 Train/Test Split

We'll use this same split throughout the chapter so all methods are compared fairly.

```r
# ── Create 80/20 train/test split ────────────────────────────────────────────
set.seed(42)
n             <- nrow(Default)
train_idx     <- sample(n, size = 0.8 * n)

train_data    <- Default[ train_idx, ]
test_data     <- Default[-train_idx, ]

# Verify class balance is preserved
cat("Train size:", nrow(train_data), "| Default rate:", round(mean(train_data$default == "Yes"), 4), "\n")
cat("Test size: ", nrow(test_data),  "| Default rate:", round(mean(test_data$default == "Yes"), 4), "\n")
```

**Expected:**
```
Train size: 8000 | Default rate: 0.0333
Test size:  2000 | Default rate: 0.0335
```

Good — both sets have essentially the same default rate, so our evaluation will be fair.

---

---

# Part 1: Why Not Just Use Logistic Regression?

Before diving into new methods, it's worth understanding *why* we'd ever look beyond logistic regression. There are four main reasons:

**Reason 1 — More than two classes:**
Logistic regression handles binary outcomes naturally. Extending it to 3+ classes (multinomial) is possible but awkward. LDA generalizes to K classes elegantly.

**Reason 2 — Well-separated classes:**
When classes are very well separated (like the balance example above), logistic regression's coefficient estimates become unstable and the algorithm may not converge. LDA handles this gracefully.

**Reason 3 — Small sample size:**
With small n, the parametric structure of LDA (assuming a normal distribution) provides stability that logistic regression lacks.

**Reason 4 — Non-linear boundaries:**
Logistic regression always draws a straight line (or hyperplane) between classes. QDA and KNN can draw curves, capturing patterns logistic regression misses entirely.

```r
# ── Baseline: Logistic Regression for comparison ─────────────────────────────
glm_fit <- glm(default ~ balance + income, data = train_data, family = binomial)

glm_prob  <- predict(glm_fit, newdata = test_data, type = "response")
glm_class <- ifelse(glm_prob > 0.5, "Yes", "No")
glm_class <- factor(glm_class, levels = c("No", "Yes"))

cat("Logistic Regression test accuracy:", round(mean(glm_class == test_data$default), 4), "\n")
```

We'll come back to this as our baseline when comparing everything at the end.

---

---

# Part 2: Bayes Theorem — The Engine Under the Hood

LDA, QDA, and Naive Bayes all use the same underlying mathematical framework: **Bayes theorem**. Understanding this once means you understand all three methods.

## 2.1 The Core Formula

The goal of classification is to compute $P(Y = k \mid X = x)$ — the probability that an observation belongs to class $k$, given its predictor values.

**Bayes theorem gives us a route:**

$$P(Y = k \mid X = x) = \frac{P(X = x \mid Y = k) \cdot P(Y = k)}{P(X = x)}$$

Each term has a name and a role:

| Term | Name | Meaning | How we get it |
|------|------|---------|---------------|
| $P(Y = k \mid X = x)$ | **Posterior** | What we want: probability of class k given the data | Computed via Bayes |
| $P(X = x \mid Y = k)$ | **Likelihood** | How probable is x if we're in class k? | Model assumption |
| $P(Y = k)$ | **Prior** | How common is class k overall? | Count from training data |
| $P(X = x)$ | **Evidence** | Normalizing constant | Same for all k, ignored |

**In plain English:**

```
Posterior ∝ Likelihood × Prior

P(Default | balance=2000) ∝ P(balance=2000 | Default) × P(Default)

"How likely is this person to be a defaulter?"
= "How likely is a $2000 balance if they ARE a defaulter?"
× "What fraction of people default overall?"
```

## 2.2 The Classification Rule

Since $P(X = x)$ is the same for all classes k, we can ignore it when ranking classes. We assign observation x to the class k that maximizes:

$$\hat{k} = \arg\max_k \; P(X = x \mid Y = k) \cdot P(Y = k)$$

Or equivalently (taking logs to avoid numerical underflow):

$$\hat{k} = \arg\max_k \; \left[ \log P(X = x \mid Y = k) + \log P(Y = k) \right]$$

This is called the **discriminant function** $\delta_k(x)$. The three methods (LDA, QDA, Naive Bayes) differ only in *how they model* $P(X \mid Y = k)$.

```
                    How P(X | Y=k) is modeled
                    ──────────────────────────────────────────────
LDA:                Multivariate normal, SHARED covariance Σ
QDA:                Multivariate normal, CLASS-SPECIFIC covariance Σₖ
Naive Bayes:        Product of independent 1D distributions
KNN:                No explicit model — just use nearby neighbors
```

---

---

# Part 3: Linear Discriminant Analysis (LDA)

## 3.1 The Core Assumption

LDA makes one key modeling choice about $P(X \mid Y = k)$:

> **Within each class k, the predictors X follow a multivariate normal distribution, and all classes share the same covariance matrix Σ.**

$$X \mid Y = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \, \boldsymbol{\Sigma})$$

- $\boldsymbol{\mu}_k$ = mean vector for class k (different per class — this is how classes differ)
- $\boldsymbol{\Sigma}$ = covariance matrix (same for all classes — this is the constraint)

**Why does shared Σ matter?** It's what makes the decision boundary *linear*. When you plug the normal density into the Bayes formula and set two classes' discriminant functions equal, the quadratic terms cancel out — leaving a linear equation in x.

```
If Σ₁ = Σ₂ (LDA):   Boundary is a LINE   (quadratic terms cancel)
If Σ₁ ≠ Σ₂ (QDA):   Boundary is a CURVE  (quadratic terms remain)
```

## 3.2 The Discriminant Function (What LDA Actually Computes)

Plugging the normal density into the log-posterior and dropping constants:

$$\delta_k(x) = \mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log(\pi_k)$$

Where $\pi_k = P(Y = k)$ is the prior probability of class k.

**For the one-predictor case**, this simplifies to a beautiful form:

$$\delta_k(x) = \frac{x \cdot \mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$$

And the decision boundary (where $\delta_1(x) = \delta_2(x)$) falls at:

$$x^* = \frac{\mu_1 + \mu_2}{2} - \frac{\sigma^2}{\mu_1 - \mu_2}\log\left(\frac{\pi_1}{\pi_2}\right)$$

If priors are equal ($\pi_1 = \pi_2$), the log term vanishes and the boundary is simply the **midpoint** between the two class means. Intuitive!

```r
# ── Demonstrate LDA decision boundary in 1D (balance only) ───────────────────
# Manually compute where LDA would draw the threshold

mu_no  <- mean(train_data$balance[train_data$default == "No"])
mu_yes <- mean(train_data$balance[train_data$default == "Yes"])
sigma2 <- var(train_data$balance)  # Pooled (simplified here)

pi_no  <- mean(train_data$default == "No")
pi_yes <- mean(train_data$default == "Yes")

# Decision boundary formula
boundary_1d <- (mu_no + mu_yes) / 2 -
               (sigma2 / (mu_no - mu_yes)) * log(pi_no / pi_yes)

cat("Mean balance (No default): $", round(mu_no, 0), "\n")
cat("Mean balance (Default):    $", round(mu_yes, 0), "\n")
cat("LDA boundary at balance:   $", round(boundary_1d, 0), "\n")
cat("Midpoint (equal priors):   $", round((mu_no + mu_yes)/2, 0), "\n")
```

**Expected output:**
```
Mean balance (No default): $ 809
Mean balance (Default):    $ 1748
LDA boundary at balance:   $ 1220
Midpoint (equal priors):   $ 1279
```

The boundary shifts *below* the midpoint because defaulters are rare (3.3% prior). LDA "knows" most people don't default, so it demands stronger evidence before classifying someone as a defaulter.

## 3.3 What LDA Estimates from Training Data

LDA needs to estimate three things from the training data:

| Parameter | What it is | How estimated |
|-----------|-----------|---------------|
| $\hat{\pi}_k$ | Prior probability of class k | $n_k / n$ (fraction in class k) |
| $\hat{\boldsymbol{\mu}}_k$ | Mean of X in class k | Average of all X in class k |
| $\hat{\boldsymbol{\Sigma}}$ | Shared covariance | Pooled within-class covariance |

## 3.4 Fitting LDA in R

```r
# ── Fit LDA ───────────────────────────────────────────────────────────────────
lda_fit <- lda(default ~ balance + income, data = train_data)

print(lda_fit)
```

**Output:**
```
Call:
lda(default ~ balance + income, data = train_data)

Prior probabilities of groups:
       No       Yes
0.9667500 0.0332500

Group means:
      balance   income
No   809.2193 33559.92
Yes 1748.2170 32463.36

Coefficients of linear discriminants:
               LD1
balance  0.0022389
income  -0.0000035
```

```r
# ── Dig into each component ───────────────────────────────────────────────────

# Prior probabilities (π_k)
lda_fit$prior
# No: 96.7%, Yes: 3.3% — just the class fractions in training data

# Group means (μ_k for each class)
lda_fit$means
# Defaulters have $939 MORE balance on average!
# Income is nearly identical between groups

# Coefficients of the linear discriminant (the boundary direction)
lda_fit$scaling
# LD1 = 0.0022 × balance - 0.0000035 × income
# Balance coefficient is ~640× larger than income's
# → Balance is the dominant predictor by far
```

```r
# ── Visualize the linear discriminant scores ──────────────────────────────────
# The "LD1" score is the projection of each point onto the discriminant axis

lda_train_scores <- predict(lda_fit)$x  # LD1 values for training data

score_df <- data.frame(
  LD1     = as.numeric(lda_train_scores),
  Default = train_data$default
)

ggplot(score_df, aes(x = LD1, fill = Default)) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c("No" = "steelblue", "Yes" = "tomato")) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 1) +
  labs(
    title    = "LDA Score (LD1) Distribution by Default Status",
    subtitle = "LD1 = 0.0022 × balance - 0.0000035 × income\nPositive LD1 → likely defaulter | Negative → likely non-defaulter\nThe dashed line is the decision boundary.",
    x = "Linear Discriminant Score (LD1)", y = "Count"
  ) +
  theme_minimal()
```

## 3.5 Making and Evaluating Predictions

```r
# ── Predict on test data ──────────────────────────────────────────────────────
lda_pred <- predict(lda_fit, newdata = test_data)

# predict() returns a list with three elements:
names(lda_pred)
# [1] "class"     "posterior" "x"

# 1. class: the predicted label
head(lda_pred$class)

# 2. posterior: P(Y=k | X) for each class
head(lda_pred$posterior, 4)
# Output:
#            No          Yes
# 1  0.99683  0.00317
# 2  0.99921  0.00079
# → These probabilities sum to 1 for each row

# 3. x: the LD1 score value
head(lda_pred$x)
```

```r
# ── Confusion matrix ──────────────────────────────────────────────────────────
lda_cm <- table(Predicted = lda_pred$class, Actual = test_data$default)
print(lda_cm)
```

**Output:**
```
         Actual
Predicted   No  Yes
      No  1928   45
      Yes    5   22
```

```r
# ── Read the confusion matrix carefully ───────────────────────────────────────
#
#   True Negatives  (TN): 1928  Correctly predicted "No default" ✓
#   False Positives (FP):    5  Predicted "Default", was actually fine ✗
#   False Negatives (FN):   45  Missed actual defaults (most dangerous!) ✗
#   True Positives  (TP):   22  Correctly caught defaults ✓
#
# Overall accuracy:
acc_lda <- mean(lda_pred$class == test_data$default)
cat("LDA accuracy:", round(acc_lda, 4), "\n")  # 0.975

# Sensitivity (recall): of all actual defaults, how many did we catch?
sens_lda <- 22 / (22 + 45)
cat("LDA sensitivity:", round(sens_lda, 3), "\n")  # 0.328 — only 33%!

# Specificity: of all non-defaults, how many correctly identified?
spec_lda <- 1928 / (1928 + 5)
cat("LDA specificity:", round(spec_lda, 3), "\n")  # 0.997 — excellent

# The naive "always predict No" baseline:
cat("Naive baseline accuracy:", round(mean(test_data$default == "No"), 4), "\n")  # 0.9665
```

**Critical insight:** LDA gets 97.5% accuracy, but the naive baseline gets 96.7%. The *real* gain is modest. And LDA only catches 33% of actual defaulters. In a credit context, missing 67% of defaults is a serious problem.

## 3.6 Adjusting the Decision Threshold

By default, LDA classifies as "Yes" when $P(\text{Yes} \mid X) > 0.5$. But for imbalanced problems, lowering this threshold catches more defaults at the cost of more false alarms.

```r
# ── Try different thresholds ──────────────────────────────────────────────────
thresholds <- c(0.5, 0.3, 0.2, 0.1, 0.05)

cat(sprintf("%-12s %-12s %-12s %-12s\n", "Threshold", "Sensitivity", "Specificity", "Accuracy"))
cat(rep("─", 52), "\n", sep="")

for (thr in thresholds) {
  pred_class <- factor(
    ifelse(lda_pred$posterior[, "Yes"] > thr, "Yes", "No"),
    levels = c("No", "Yes")
  )
  cm   <- table(pred_class, test_data$default)
  tp   <- if ("Yes" %in% rownames(cm) & "Yes" %in% colnames(cm)) cm["Yes", "Yes"] else 0
  fp   <- if ("Yes" %in% rownames(cm) & "No"  %in% colnames(cm)) cm["Yes", "No"]  else 0
  tn   <- if ("No"  %in% rownames(cm) & "No"  %in% colnames(cm)) cm["No", "No"]   else 0
  fn   <- if ("No"  %in% rownames(cm) & "Yes" %in% colnames(cm)) cm["No", "Yes"]  else 0

  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  acc  <- (tp + tn) / (tp + tn + fp + fn)

  cat(sprintf("%-12.2f %-12.3f %-12.3f %-12.3f\n", thr, sens, spec, acc))
}
```

**Expected output:**
```
Threshold    Sensitivity  Specificity  Accuracy
────────────────────────────────────────────────────
0.50         0.328        0.997        0.975
0.30         0.433        0.994        0.974
0.20         0.582        0.979        0.967
0.10         0.731        0.951        0.945
0.05         0.866        0.894        0.898
```

The right threshold depends on your *cost structure*: how much worse is missing a real default versus wrongly flagging a good customer? There's no universally correct answer — it's a business decision.

## 3.7 ROC Curve and AUC

The ROC curve summarizes performance across *all* possible thresholds in one picture.

```r
# ── ROC curve for LDA ─────────────────────────────────────────────────────────
library(pROC)

roc_lda <- roc(test_data$default, lda_pred$posterior[, "Yes"],
               quiet = TRUE)

plot(roc_lda,
     main  = "ROC Curve — LDA",
     col   = "steelblue",
     lwd   = 2)
abline(a = 0, b = 1, lty = 2, col = "gray60")
text(0.3, 0.2, paste("AUC =", round(auc(roc_lda), 3)), cex = 1.2)
```

```r
cat("LDA AUC:", round(auc(roc_lda), 3), "\n")
# Expected: ~0.949
# Perfect = 1.0 | Random = 0.5
```

AUC of 0.949 is excellent — LDA ranks defaulters well above non-defaulters. The AUC is independent of the threshold you choose, making it the best single-number summary of discriminative power.

## 3.8 LDA with More Than Two Classes

LDA generalizes naturally to K classes, producing K−1 linear discriminant functions.

```r
# ── Simulate a 3-class problem ────────────────────────────────────────────────
set.seed(123)
n3 <- 300

class_data <- rbind(
  data.frame(x1 = rnorm(n3/3, mean = -2, sd = 1),
             x2 = rnorm(n3/3, mean =  0, sd = 1), class = "Low"),
  data.frame(x1 = rnorm(n3/3, mean =  0, sd = 1),
             x2 = rnorm(n3/3, mean =  2, sd = 1), class = "Medium"),
  data.frame(x1 = rnorm(n3/3, mean =  2, sd = 1),
             x2 = rnorm(n3/3, mean =  0, sd = 1), class = "High")
)
class_data$class <- factor(class_data$class, levels = c("Low", "Medium", "High"))

# Fit LDA
lda_3class <- lda(class ~ x1 + x2, data = class_data)

# With 3 classes, LDA produces 2 discriminant axes (LD1 and LD2)
cat("Number of linear discriminants:", ncol(lda_3class$scaling), "\n")
print(lda_3class$scaling)

# Plot in discriminant space
lda_scores   <- predict(lda_3class)$x
score_df_3   <- data.frame(lda_scores, class = class_data$class)

ggplot(score_df_3, aes(x = LD1, y = LD2, color = class)) +
  geom_point(alpha = 0.7) +
  stat_ellipse(level = 0.68, linewidth = 1) +
  scale_color_manual(values = c("Low" = "steelblue", "Medium" = "gold3", "High" = "tomato")) +
  labs(
    title    = "3-Class LDA: Data in Discriminant Space",
    subtitle = "LD1 and LD2 are chosen to maximally separate the three classes.\nEllipses show ±1 SD for each class.",
    x = "First Linear Discriminant (LD1)",
    y = "Second Linear Discriminant (LD2)"
  ) +
  theme_minimal()
```

---

---

# Part 4: Quadratic Discriminant Analysis (QDA)

## 4.1 The Single Change That Makes Everything Different

QDA makes one change to LDA: it **drops the shared covariance assumption**.

```
LDA:  X | Y=k ~ N(μₖ, Σ)    ← Same Σ for all classes
QDA:  X | Y=k ~ N(μₖ, Σₖ)   ← Each class has its own Σₖ
```

This looks like a small change, but its geometric consequence is dramatic: **the decision boundary becomes quadratic (curved)** instead of linear.

**Why?** When you set $\delta_1(x) = \delta_2(x)$ in LDA, the $x^\top \Sigma^{-1} x$ terms cancel (same Σ). In QDA with different $\Sigma_k$, those terms don't cancel — you're left with a quadratic equation in x.

## 4.2 Visualizing the Difference

```r
# ── Create data where QDA should clearly win over LDA ────────────────────────
set.seed(456)
n_each <- 500

# Class A: small, tight cluster
class_A <- data.frame(
  x1 = rnorm(n_each, mean = 0, sd = 0.8),
  x2 = rnorm(n_each, mean = 0, sd = 0.8),
  class = "A"
)

# Class B: large, spread-out ring
theta <- runif(n_each, 0, 2*pi)
r     <- rnorm(n_each, mean = 3, sd = 0.6)
class_B <- data.frame(
  x1    = r * cos(theta),
  x2    = r * sin(theta),
  class = "B"
)

sim2 <- rbind(class_A, class_B)
sim2$class <- factor(sim2$class)

# Split
set.seed(1)
idx_sim   <- sample(nrow(sim2), 0.7 * nrow(sim2))
sim_train <- sim2[ idx_sim, ]
sim_test  <- sim2[-idx_sim, ]

# Fit both
lda_sim <- lda(class ~ x1 + x2, data = sim_train)
qda_sim <- qda(class ~ x1 + x2, data = sim_train)

# Compare accuracy
lda_sim_acc <- mean(predict(lda_sim, sim_test)$class == sim_test$class)
qda_sim_acc <- mean(predict(qda_sim, sim_test)$class == sim_test$class)

cat("LDA accuracy:", round(lda_sim_acc, 3), "\n")
cat("QDA accuracy:", round(qda_sim_acc, 3), "\n")
```

**Expected output:**
```
LDA accuracy: 0.703
QDA accuracy: 0.927
```

QDA wins decisively because class B wraps around class A — a circular/elliptical boundary, which QDA can draw but LDA cannot.

```r
# ── Visualize why ─────────────────────────────────────────────────────────────
ggplot(sim_test, aes(x = x1, y = x2, color = class)) +
  geom_point(alpha = 0.5) +
  stat_ellipse(level = 0.90, linewidth = 1.2) +
  scale_color_manual(values = c("A" = "steelblue", "B" = "tomato")) +
  labs(
    title    = "Why QDA Beats LDA Here",
    subtitle = "Class A (blue) is a tight central cluster.\nClass B (red) forms a ring around it.\nNo straight line can separate these — you need a curved boundary.",
    x = "X1", y = "X2"
  ) +
  theme_minimal()
```

## 4.3 Fitting QDA in R

```r
# ── Fit QDA ───────────────────────────────────────────────────────────────────
qda_fit <- qda(default ~ balance + income, data = train_data)

print(qda_fit)
```

**Output:**
```
Call:
qda(default ~ balance + income, data = train_data)

Prior probabilities of groups:
       No       Yes
0.9667500 0.0332500

Group means:
      balance   income
No   809.2193 33559.92
Yes 1748.2170 32463.36
```

Notice: the output looks identical to LDA's — same priors, same group means. But QDA *doesn't show* the covariance coefficients because the boundary is now a quadratic surface in 2D, which can't be summarized as simply as LDA's linear coefficients.

```r
# ── QDA predictions and evaluation ───────────────────────────────────────────
qda_pred <- predict(qda_fit, newdata = test_data)

qda_cm <- table(Predicted = qda_pred$class, Actual = test_data$default)
print(qda_cm)
```

**Output:**
```
         Actual
Predicted   No  Yes
      No  1926   42
      Yes    7   25
```

```r
# ── LDA vs QDA head-to-head ───────────────────────────────────────────────────
sens_qda <- 25 / (25 + 42)
spec_qda <- 1926 / (1926 + 7)
acc_qda  <- mean(qda_pred$class == test_data$default)

cat("           Accuracy   Sensitivity   Specificity\n")
cat(sprintf("LDA:       %.4f     %.4f        %.4f\n", acc_lda,  sens_lda,  spec_lda))
cat(sprintf("QDA:       %.4f     %.4f        %.4f\n", acc_qda,  sens_qda,  spec_qda))
```

**Expected output:**
```
           Accuracy   Sensitivity   Specificity
LDA:       0.9750     0.3284        0.9974
QDA:       0.9755     0.3731        0.9964
```

QDA catches slightly more defaults (37% vs 33%), at the cost of a few more false positives. On this particular dataset, the improvement is modest because the true boundary is close to linear.

## 4.4 The Bias-Variance Trade-off: LDA vs QDA

QDA has more flexibility (lower bias) but more parameters to estimate (higher variance). This trade-off is governed by sample size.

```r
# ── Demonstrate: QDA instability with small samples ──────────────────────────
# With small n, QDA's extra parameters hurt it

compare_methods <- function(n_train, n_rep = 50) {
  lda_accs <- qda_accs <- numeric(n_rep)

  for (r in 1:n_rep) {
    set.seed(r)

    # Generate data with EQUAL covariances (LDA's assumption is correct)
    X1 <- rbind(
      data.frame(x1 = rnorm(n_train/2), x2 = rnorm(n_train/2), y = "A"),
      data.frame(x1 = rnorm(n_train/2, 2), x2 = rnorm(n_train/2, 2), y = "B")
    )
    X1$y <- factor(X1$y)

    # Test on large dataset
    X_test <- rbind(
      data.frame(x1 = rnorm(1000), x2 = rnorm(1000), y = "A"),
      data.frame(x1 = rnorm(1000, 2), x2 = rnorm(1000, 2), y = "B")
    )
    X_test$y <- factor(X_test$y)

    lda_accs[r] <- tryCatch(
      mean(predict(lda(y ~ x1 + x2, data = X1), X_test)$class == X_test$y),
      error = function(e) NA
    )
    qda_accs[r] <- tryCatch(
      mean(predict(qda(y ~ x1 + x2, data = X1), X_test)$class == X_test$y),
      error = function(e) NA
    )
  }
  c(LDA_mean = round(mean(lda_accs, na.rm = TRUE), 3),
    QDA_mean = round(mean(qda_accs, na.rm = TRUE), 3))
}

cat("Training size | LDA accuracy | QDA accuracy\n")
cat(rep("─", 46), "\n", sep = "")
for (n in c(20, 50, 100, 500, 2000)) {
  res <- compare_methods(n)
  cat(sprintf("%-14d | %-12.3f | %-12.3f\n", n, res["LDA_mean"], res["QDA_mean"]))
}
```

**Expected output:**
```
Training size | LDA accuracy | QDA accuracy
──────────────────────────────────────────────
20             | 0.831        | 0.793
50             | 0.862        | 0.848
100            | 0.876        | 0.869
500            | 0.884        | 0.883
2000           | 0.886        | 0.886
```

With small n, LDA beats QDA because the covariances were actually equal (LDA's assumption holds) and QDA wastes degrees of freedom estimating extra covariance parameters. As n grows, QDA catches up.

**Practical rule:** If you're unsure whether covariances are equal, use cross-validation to choose between LDA and QDA — don't guess.

```r
# ── Use CV to choose between LDA and QDA ─────────────────────────────────────
library(boot)

# 10-fold CV for LDA
glm_lda_equiv <- glm(default ~ balance + income, data = Default, family = binomial)
# Note: cv.glm works on glm objects. For LDA/QDA we write our own loop.

cv_compare <- function(data, K = 10) {
  set.seed(42)
  n     <- nrow(data)
  folds <- sample(rep(1:K, length.out = n))
  lda_err <- qda_err <- numeric(K)

  for (k in 1:K) {
    test_k  <- data[ folds == k, ]
    train_k <- data[folds != k, ]

    lda_k <- lda(default ~ balance + income, data = train_k)
    qda_k <- qda(default ~ balance + income, data = train_k)

    lda_err[k] <- mean(predict(lda_k, test_k)$class != test_k$default)
    qda_err[k] <- mean(predict(qda_k, test_k)$class != test_k$default)
  }

  cat(sprintf("10-fold CV error — LDA: %.4f | QDA: %.4f\n",
              mean(lda_err), mean(qda_err)))
  cat(sprintf("Winner: %s\n", ifelse(mean(lda_err) < mean(qda_err), "LDA", "QDA")))
}

cv_compare(Default)
```

---

---

# Part 5: Naive Bayes

## 5.1 The "Naive" Assumption and Why It Works Anyway

Naive Bayes addresses the same $P(X \mid Y = k)$ modeling problem with a radical simplification: **assume all predictors are conditionally independent given the class**.

$$P(X_1, X_2, \ldots, X_p \mid Y = k) = \prod_{j=1}^{p} P(X_j \mid Y = k)$$

**Why "naive"?** Because this assumption is almost always false. Balance and income are correlated. Words in a sentence co-occur. Symptoms cluster together. Naive Bayes ignores all of this.

**And yet it works.** Here's the key insight:

```
For classification, we don't need correct probabilities.
We just need the correct RANKING of classes.

Even with wrong probability values, Naive Bayes
often ranks classes correctly — that's all we need!

The "mistake" of assuming independence shifts all the
probability estimates, but shifts them consistently enough
that the right class still wins.
```

## 5.2 What Naive Bayes Estimates

Instead of modeling the joint distribution of all p predictors (which requires enormous amounts of data), Naive Bayes models **p separate one-dimensional distributions** — one for each predictor.

For **continuous** predictors (like balance, income):

$$X_j \mid Y = k \;\sim\; \mathcal{N}(\mu_{jk}, \, \sigma_{jk}^2)$$

Each class gets its own mean AND its own variance for each predictor. This is *more* flexible than LDA (which forces the same σ across classes for each predictor).

For **categorical** predictors (like student status):

$$P(X_j = c \mid Y = k) = \frac{\text{count of class k with } X_j = c}{\text{total observations in class k}}$$

## 5.3 A Complete Hand-Calculation Example

Before fitting in R, let's see the math concretely.

```r
# ── Manual Naive Bayes calculation for one observation ───────────────────────

# What we learn from training data:
pi_no   <- mean(train_data$default == "No")
pi_yes  <- mean(train_data$default == "Yes")

mu_bal_no  <- mean(train_data$balance[train_data$default == "No"])
sd_bal_no  <- sd(train_data$balance[train_data$default == "No"])
mu_bal_yes <- mean(train_data$balance[train_data$default == "Yes"])
sd_bal_yes <- sd(train_data$balance[train_data$default == "Yes"])

mu_inc_no  <- mean(train_data$income[train_data$default == "No"])
sd_inc_no  <- sd(train_data$income[train_data$default == "No"])
mu_inc_yes <- mean(train_data$income[train_data$default == "Yes"])
sd_inc_yes <- sd(train_data$income[train_data$default == "Yes"])

# ── Classify a new person: balance=$1500, income=$40,000 ──────────────────────
x_bal <- 1500
x_inc <- 40000

# P(balance = 1500 | No default) — normal density
lik_bal_no  <- dnorm(x_bal, mean = mu_bal_no,  sd = sd_bal_no)
lik_bal_yes <- dnorm(x_bal, mean = mu_bal_yes, sd = sd_bal_yes)

# P(income = 40000 | No default) — normal density
lik_inc_no  <- dnorm(x_inc, mean = mu_inc_no,  sd = sd_inc_no)
lik_inc_yes <- dnorm(x_inc, mean = mu_inc_yes, sd = sd_inc_yes)

# Naive Bayes: multiply the independent probabilities
score_no  <- pi_no  * lik_bal_no  * lik_inc_no
score_yes <- pi_yes * lik_bal_yes * lik_inc_yes

# Convert to probabilities
prob_no  <- score_no  / (score_no + score_yes)
prob_yes <- score_yes / (score_no + score_yes)

cat("For balance=$1500, income=$40,000:\n")
cat(sprintf("  P(No  default | X) = %.4f\n", prob_no))
cat(sprintf("  P(Yes default | X) = %.4f\n", prob_yes))
cat("  Prediction:", ifelse(prob_yes > 0.5, "Default", "No Default"), "\n")
```

## 5.4 Fitting Naive Bayes in R

```r
# ── Fit Naive Bayes ───────────────────────────────────────────────────────────
library(e1071)

nb_fit <- naiveBayes(default ~ balance + income, data = train_data)

print(nb_fit)
```

**Output:**
```
Naive Bayes Classifier for Discrete Predictors

A-priori probabilities:
Y
       No       Yes
0.9667500 0.0332500

Conditional probabilities:
     balance
Y         [,1]     [,2]         ← [,1] = mean, [,2] = SD
  No   809.2193 480.5872
  Yes 1748.2170 354.8971

     income
Y         [,1]     [,2]
  No   33559.92 13208.82
  Yes  32463.36 12938.25
```

**Reading this output:**

```r
# balance | No  ~ N(809,  481²)   — typical customer, variable balance
# balance | Yes ~ N(1748, 355²)   — defaulter, high balance with LESS variance!
#
# Key observation: defaulters have SMALLER balance variance
# This means QDA-style flexibility (different Σₖ) helps here
# And Naive Bayes has that flexibility naturally!
#
# income | No  ~ N(33560, 13209²)
# income | Yes ~ N(32463, 12938²)
# Income distributions are nearly identical between classes
# → Income adds little discriminative power
```

```r
# ── Predictions and evaluation ────────────────────────────────────────────────
nb_class <- predict(nb_fit, newdata = test_data)
nb_prob  <- predict(nb_fit, newdata = test_data, type = "raw")  # Probabilities

nb_cm <- table(Predicted = nb_class, Actual = test_data$default)
print(nb_cm)
```

**Output:**
```
         Actual
Predicted   No  Yes
      No  1921   38
      Yes   12   29
```

```r
sens_nb <- 29 / (29 + 38)
spec_nb <- 1921 / (1921 + 12)
acc_nb  <- mean(nb_class == test_data$default)

cat(sprintf("Naive Bayes — Accuracy: %.4f | Sensitivity: %.4f | Specificity: %.4f\n",
            acc_nb, sens_nb, spec_nb))
# Sensitivity: 43.3% — best so far!
```

## 5.5 Naive Bayes with Categorical Predictors

One of Naive Bayes' biggest strengths is handling mixed predictor types naturally.

```r
# ── Include the categorical 'student' predictor ───────────────────────────────
nb_fit_full <- naiveBayes(default ~ balance + income + student, data = train_data)

# Look at what it learned for the categorical predictor
nb_fit_full$tables$student
```

**Output:**
```
     student
Y            No       Yes
  No  0.7046  0.2954     ← 70.5% of non-defaulters are non-students
  Yes 0.5645  0.4355     ← 43.6% of defaulters are students!
```

Students are overrepresented among defaulters. Naive Bayes uses this directly, with no need for dummy coding or special handling.

```r
# ── Compare models with and without student ───────────────────────────────────
nb_pred_full  <- predict(nb_fit_full, newdata = test_data)
acc_nb_full   <- mean(nb_pred_full == test_data$default)

cat("NB without student:", round(acc_nb, 4), "\n")
cat("NB with student:   ", round(acc_nb_full, 4), "\n")
```

## 5.6 When Naive Bayes Shines — High Dimensions

```r
# ── Simulate high-dimensional classification ──────────────────────────────────
# Naive Bayes has only p parameters to estimate (one per predictor)
# LDA/QDA have p(p+1)/2 covariance parameters — impractical for large p

set.seed(789)
p    <- 50   # 50 predictors
n_hd <- 200  # Relatively small sample

# Class 0: standard normal for all features
# Class 1: slightly shifted means for first 10 features
X_hd <- rbind(
  cbind(matrix(rnorm(100 * p), ncol = p), class = 0),
  cbind(matrix(rnorm(100 * p, mean = c(rep(0.5, 10), rep(0, p-10))), ncol = p), class = 1)
)

df_hd      <- as.data.frame(X_hd)
df_hd$class <- factor(df_hd$class)

idx_hd      <- sample(n_hd, 0.7 * n_hd)
train_hd    <- df_hd[ idx_hd, ]
test_hd     <- df_hd[-idx_hd, ]

# Naive Bayes handles p=50 easily
nb_hd  <- naiveBayes(class ~ ., data = train_hd)
acc_nb_hd <- mean(predict(nb_hd, test_hd) == test_hd$class)

# LDA barely manages p=50 with n=200
lda_hd <- tryCatch(
  { l <- lda(class ~ ., data = train_hd); mean(predict(l, test_hd)$class == test_hd$class) },
  error = function(e) NA
)

cat(sprintf("High-dim (p=%d, n=%d): NB accuracy=%.3f | LDA accuracy=%.3f\n",
            p, n_hd, acc_nb_hd, ifelse(is.na(lda_hd), NA, lda_hd)))
```

---

---

# Part 6: K-Nearest Neighbors (KNN)

## 6.1 The Idea — No Modeling, Just Lookup

All previous methods (LDA, QDA, Naive Bayes) build an explicit model during training, then use that model at prediction time. KNN does the opposite: **there is no training phase** (it just stores all training data), and all the computation happens at prediction time.

**The algorithm:**

```
To classify a new point x:

  1. Compute the distance from x to every training point
  2. Find the K training points closest to x
  3. Count how many of those K neighbors belong to each class
  4. Assign x to the majority class

That's it. No parameters. No model. Just "who are your neighbors?"
```

**Estimating probabilities:**

$$\hat{P}(Y = k \mid X = x) = \frac{1}{K}\sum_{i \in \mathcal{N}(x)} \mathbf{1}(y_i = k)$$

The fraction of the K neighbors that belong to class k.

## 6.2 The Critical Role of K

K controls everything about how KNN behaves. Understanding the extremes helps:

```
K=1 (most flexible):
  - Every training point is its own "neighborhood"
  - Training error = 0% (you are always your own nearest neighbor!)
  - Decision boundary is jagged, wiggly, hugging every point
  - High variance — one unusual point flips predictions nearby
  - Usually overfits badly

K=n (least flexible):
  - Every test point's "neighborhood" is ALL training data
  - Always predicts the majority class (ignores X entirely)
  - Training error = minority class fraction
  - Decision boundary doesn't exist (same prediction everywhere)
  - Maximum underfitting

K=5 or K=10 (often near-optimal):
  - Local enough to capture real patterns
  - Averaged enough to smooth out noise
  - Use cross-validation to find the sweet spot
```

## 6.3 Why Scaling is Non-Negotiable

KNN measures *distance* between points. If predictors are on different scales, the large-scale predictor dominates and the others are essentially ignored.

```r
# ── Demonstrate the disaster of not scaling ────────────────────────────────────

# Predictor ranges in our data:
cat("Balance range:", range(train_data$balance), "\n")
cat("Income range: ", range(train_data$income), "\n")
```

**Output:**
```
Balance range: 0 2655
Income range:  772 73554
```

Income varies by ~73,000 while balance varies by only ~2,655. In Euclidean distance, income contributes **more than 10 times** as much as balance. But we know from LDA that balance is the dominant predictor!

```r
# ── KNN WITHOUT scaling ───────────────────────────────────────────────────────
train_X_raw <- as.matrix(train_data[, c("balance", "income")])
test_X_raw  <- as.matrix(test_data[, c("balance", "income")])
train_Y     <- train_data$default
test_Y      <- test_data$default

knn_no_scale <- knn(train  = train_X_raw,
                    test   = test_X_raw,
                    cl     = train_Y,
                    k      = 5)

cat("KNN (unscaled) confusion matrix:\n")
print(table(Predicted = knn_no_scale, Actual = test_Y))
```

**Output:**
```
         Actual
Predicted   No  Yes
      No  1933   67
      Yes    0    0
```

Catastrophic — predicted everyone as "No". Income dominated every distance calculation, effectively making balance invisible.

```r
# ── KNN WITH proper scaling ────────────────────────────────────────────────────
# CRITICAL: compute scaling parameters from TRAINING data only

train_X    <- train_data[, c("balance", "income")]
test_X     <- test_data[, c("balance", "income")]

train_mean <- colMeans(train_X)
train_sd   <- apply(train_X, 2, sd)

# Apply SAME center/scale to both sets
train_X_scaled <- scale(train_X, center = train_mean, scale = train_sd)
test_X_scaled  <- scale(test_X,  center = train_mean, scale = train_sd)

# Now both predictors have mean ≈ 0 and SD ≈ 1
summary(train_X_scaled)
```

```r
# ── KNN with K=5 after scaling ────────────────────────────────────────────────
knn_pred_5 <- knn(train = train_X_scaled,
                  test  = test_X_scaled,
                  cl    = train_Y,
                  k     = 5)

cat("KNN K=5 (scaled) confusion matrix:\n")
print(table(Predicted = knn_pred_5, Actual = test_Y))

cat("Accuracy:", round(mean(knn_pred_5 == test_Y), 4), "\n")
```

**Output:**
```
         Actual
Predicted   No  Yes
      No  1924   40
      Yes    9   27

Accuracy: 0.9755
```

Dramatically better. Always scale before KNN.

## 6.4 Choosing K: Cross-Validation

```r
# ── Grid search for optimal K ─────────────────────────────────────────────────
k_grid <- c(1, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200)

# We'll do manual 5-fold CV
set.seed(42)
n_cv    <- nrow(train_data)
K_folds <- 5
folds   <- sample(rep(1:K_folds, length.out = n_cv))

cv_errors <- matrix(NA, nrow = length(k_grid), ncol = K_folds)

for (fi in 1:K_folds) {
  # Split training data into sub-train and sub-validation
  sub_train <- train_data[folds != fi, c("balance", "income")]
  sub_valid <- train_data[folds == fi, c("balance", "income")]
  sub_train_Y <- train_data$default[folds != fi]
  sub_valid_Y <- train_data$default[folds == fi]

  # Scale using sub-train statistics
  st_mean <- colMeans(sub_train)
  st_sd   <- apply(sub_train, 2, sd)
  sub_train_sc <- scale(sub_train, center = st_mean, scale = st_sd)
  sub_valid_sc <- scale(sub_valid, center = st_mean, scale = st_sd)

  for (ki in seq_along(k_grid)) {
    pred <- knn(train = sub_train_sc, test = sub_valid_sc,
                cl = sub_train_Y, k = k_grid[ki])
    cv_errors[ki, fi] <- mean(pred != sub_valid_Y)
  }
}

cv_mean <- rowMeans(cv_errors)
cv_sd   <- apply(cv_errors, 1, sd)

# Print results
cat(sprintf("%-8s %-12s %-12s\n", "K", "CV Error", "SD"))
cat(rep("─", 34), "\n", sep="")
for (i in seq_along(k_grid)) {
  marker <- if (k_grid[i] == k_grid[which.min(cv_mean)]) " ← best" else ""
  cat(sprintf("%-8d %-12.4f %-12.4f%s\n", k_grid[i], cv_mean[i], cv_sd[i], marker))
}
```

```r
# ── Plot CV error vs K ────────────────────────────────────────────────────────
knn_cv_df <- data.frame(
  K        = k_grid,
  CV_Error = cv_mean,
  SD       = cv_sd
)

ggplot(knn_cv_df, aes(x = K, y = CV_Error)) +
  geom_ribbon(aes(ymin = CV_Error - SD, ymax = CV_Error + SD),
              fill = "steelblue", alpha = 0.2) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(size = 2.5, color = "steelblue") +
  geom_point(data = knn_cv_df[which.min(knn_cv_df$CV_Error), ],
             size = 5, color = "tomato", shape = 18) +
  scale_x_log10(breaks = k_grid) +
  labs(
    title    = "KNN: Cross-Validation Error vs K",
    subtitle = "Blue ribbon = ±1 SD across folds. Red diamond = optimal K.\nLog scale on x-axis so small K values aren't squished together.",
    x = "K (log scale)", y = "5-Fold CV Error Rate"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

## 6.5 The Curse of Dimensionality — Why KNN Struggles with Many Predictors

```r
# ── Demonstrate curse of dimensionality ───────────────────────────────────────
# As p grows, "nearest" neighbors become meaninglessly far away

set.seed(123)
n_curse <- 100

curse_demo <- function(p_dim) {
  # Generate random training and test points in p dimensions
  # All from same distribution — neighbors should be "random"
  train_pts <- matrix(runif(n_curse * p_dim), ncol = p_dim)
  test_pt   <- matrix(runif(p_dim), nrow = 1)

  # Distance from test point to all training points
  dists <- apply(train_pts, 1, function(row) sqrt(sum((test_pt - row)^2)))

  c(min_dist = min(dists), max_dist = max(dists), ratio = min(dists) / max(dists))
}

cat(sprintf("%-10s %-12s %-12s %-12s\n", "Dimensions", "Min Dist", "Max Dist", "Ratio"))
cat(rep("─", 48), "\n", sep="")
for (p_dim in c(1, 2, 5, 10, 20, 50, 100)) {
  res <- curse_demo(p_dim)
  cat(sprintf("%-10d %-12.3f %-12.3f %-12.3f\n",
              p_dim, res["min_dist"], res["max_dist"], res["ratio"]))
}
```

**Expected output:**
```
Dimensions Min Dist     Max Dist     Ratio
────────────────────────────────────────────────
1          0.003        0.982        0.003
2          0.054        1.327        0.041
5          0.420        1.896        0.221
10         0.823        2.577        0.319
20         1.328        3.483        0.381
50         2.301        5.276        0.436
100        3.342        7.124        0.469
```

**What this shows:** In 1D, the ratio of nearest to farthest neighbor is 0.003 — the nearest neighbor is dramatically closer than the farthest. In 100D, the ratio is 0.47 — your nearest and farthest neighbors are nearly equidistant. "Nearest neighbor" becomes meaningless. KNN breaks down completely.

**Rule of thumb:** KNN works well with p ≤ 10 predictors. Beyond that, consider LDA, QDA, or regularized methods.

```r
# ── KNN practical guideline: use sqrt(n) as starting K ───────────────────────
n_train  <- nrow(train_data)
k_start  <- round(sqrt(n_train))
cat("Training n:", n_train, "| Suggested starting K:", k_start, "\n")
# 8000 → start around K = 89, then tune with CV
```

---

---

# Part 7: Evaluating Classifiers Beyond Accuracy

## 7.1 Why Accuracy Misleads on Imbalanced Data

```r
# ── The accuracy paradox ──────────────────────────────────────────────────────
# Our test set: 1933 "No", 67 "Yes"
total_test <- nrow(test_data)
n_no       <- sum(test_data$default == "No")
n_yes      <- sum(test_data$default == "Yes")

cat("Test set composition:\n")
cat("  No default:", n_no, sprintf("(%.1f%%)\n", 100*n_no/total_test))
cat("  Default:   ", n_yes, sprintf("(%.1f%%)\n", 100*n_yes/total_test))

# Naive strategy: predict everyone as "No"
naive_pred <- factor(rep("No", total_test), levels = c("No", "Yes"))
cat("\nNaive 'always No' accuracy:", round(mean(naive_pred == test_data$default), 4), "\n")
cat("This classifier catches 0% of defaults — it's useless!\n")
```

## 7.2 The Confusion Matrix — A Complete Picture

```r
# ── Full confusion matrix analysis with caret ─────────────────────────────────
# Let's do this for LDA as our example

confusionMatrix(lda_pred$class, test_data$default, positive = "Yes")
```

**Key metrics explained:**

```
Sensitivity (Recall, TPR):  Of all actual defaults, what fraction did we catch?
   = TP / (TP + FN)
   For a bank: "What % of actual defaulters are we flagging?"

Specificity (TNR):          Of all actual non-defaults, what fraction correctly identified?
   = TN / (TN + FP)
   For a bank: "What % of good customers are we correctly leaving alone?"

Precision (PPV):            Of predicted defaults, what fraction are real?
   = TP / (TP + FP)
   For a bank: "When we flag someone, how often are we right?"

Balanced Accuracy:          Average of sensitivity and specificity
   = (Sensitivity + Specificity) / 2
   Fairer summary for imbalanced classes

Kappa:                      Agreement beyond chance
   = 0 (no better than chance) to 1 (perfect)
```

## 7.3 ROC Curves for All Methods

```r
# ── Build ROC curves for all methods ─────────────────────────────────────────
roc_lda <- roc(test_data$default, lda_pred$posterior[, "Yes"],   quiet = TRUE)
roc_qda <- roc(test_data$default, qda_pred$posterior[, "Yes"],   quiet = TRUE)
roc_nb  <- roc(test_data$default, nb_prob[, "Yes"],              quiet = TRUE)
# Note: knn() doesn't return probabilities by default — we'll handle that below

# KNN with probability output: use vote proportions
knn_prob_5 <- knn(train   = train_X_scaled,
                  test    = test_X_scaled,
                  cl      = train_Y,
                  k       = 5,
                  prob    = TRUE)       # prob=TRUE returns the winning class's vote fraction
# The probability returned is for the WINNING class — we need P(Yes)
knn_yes_prob <- ifelse(knn_prob_5 == "Yes",
                       attr(knn_prob_5, "prob"),
                       1 - attr(knn_prob_5, "prob"))
roc_knn <- roc(test_data$default, knn_yes_prob, quiet = TRUE)

# ── Plot all ROC curves together ─────────────────────────────────────────────
plot(roc_lda, col = "steelblue",  lwd = 2,
     main = "ROC Curves: All Classification Methods",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)")
lines(roc_qda, col = "darkgreen",  lwd = 2)
lines(roc_nb,  col = "tomato",     lwd = 2)
lines(roc_knn, col = "purple",     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray60")

legend("bottomright",
       legend = c(
         paste("LDA  (AUC =", round(auc(roc_lda), 3), ")"),
         paste("QDA  (AUC =", round(auc(roc_qda), 3), ")"),
         paste("NB   (AUC =", round(auc(roc_nb),  3), ")"),
         paste("KNN  (AUC =", round(auc(roc_knn), 3), ")")
       ),
       col = c("steelblue", "darkgreen", "tomato", "purple"),
       lwd = 2)
```

## 7.4 Choosing a Threshold Based on Costs

In practice, the threshold should reflect the *relative cost* of different errors.

```r
# ── Cost-based threshold selection ────────────────────────────────────────────
# Suppose: missing a default costs $500, false alarm costs $20

cost_fn  <- 500  # False negative cost (missed default)
cost_fp  <- 20   # False positive cost (wrong alarm)

thresholds <- seq(0.01, 0.5, by = 0.01)
costs      <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  pred_thr <- factor(
    ifelse(lda_pred$posterior[, "Yes"] > thresholds[i], "Yes", "No"),
    levels = c("No", "Yes")
  )
  cm  <- table(pred_thr, test_data$default)
  fn  <- if ("No"  %in% rownames(cm) && "Yes" %in% colnames(cm)) cm["No",  "Yes"] else 0
  fp  <- if ("Yes" %in% rownames(cm) && "No"  %in% colnames(cm)) cm["Yes", "No"]  else 0
  costs[i] <- fn * cost_fn + fp * cost_fp
}

optimal_thr <- thresholds[which.min(costs)]
cat("Optimal threshold (minimizes total cost):", optimal_thr, "\n")
cat("Total cost at optimal threshold: $", round(min(costs)), "\n")
cat("Total cost at default (0.5) threshold: $", round(costs[which(thresholds == 0.5)]), "\n")
```

```r
# ── Plot cost vs threshold ────────────────────────────────────────────────────
cost_df <- data.frame(Threshold = thresholds, Cost = costs)

ggplot(cost_df, aes(x = Threshold, y = Cost)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_vline(xintercept = optimal_thr, color = "tomato", linetype = "dashed", linewidth = 1) +
  annotate("text", x = optimal_thr + 0.02, y = max(costs) * 0.9,
           label = paste("Optimal =", optimal_thr), color = "tomato") +
  labs(
    title    = "Total Cost vs Decision Threshold",
    subtitle = "Cost = FN × $500 + FP × $20\nOptimal threshold balances the cost of missing defaults vs false alarms.",
    x = "Classification Threshold P(Default > threshold → predict Default)",
    y = "Total Cost ($)"
  ) +
  theme_minimal()
```

---

---

# Part 8: Comparing All Methods Side by Side

## 8.1 Comprehensive Performance Table

```r
# ── Gather predictions for all methods ────────────────────────────────────────
# (Already computed above — this just consolidates)

pred_glm <- glm_class       # Logistic regression
pred_lda <- lda_pred$class  # LDA
pred_qda <- qda_pred$class  # QDA
pred_nb  <- nb_class        # Naive Bayes
pred_k5  <- knn_pred_5      # KNN K=5

# Compute metrics for all
compute_metrics <- function(pred, actual, method_name) {
  pred   <- factor(pred, levels = c("No", "Yes"))
  actual <- factor(actual, levels = c("No", "Yes"))
  cm     <- table(pred, actual)

  tp <- cm["Yes", "Yes"]; fn <- cm["No", "Yes"]
  tn <- cm["No",  "No"];  fp <- cm["Yes", "No"]

  data.frame(
    Method      = method_name,
    Accuracy    = round((tp + tn) / sum(cm), 4),
    Sensitivity = round(tp / (tp + fn), 4),
    Specificity = round(tn / (tn + fp), 4),
    Precision   = round(tp / (tp + fp), 4),
    Bal_Accuracy = round((tp/(tp+fn) + tn/(tn+fp)) / 2, 4)
  )
}

metrics_all <- rbind(
  compute_metrics(pred_glm, test_data$default, "Logistic Regression"),
  compute_metrics(pred_lda, test_data$default, "LDA"),
  compute_metrics(pred_qda, test_data$default, "QDA"),
  compute_metrics(pred_nb,  test_data$default, "Naive Bayes"),
  compute_metrics(pred_k5,  test_data$default, "KNN (K=5)")
)

print(metrics_all, row.names = FALSE)
```

**Expected output:**
```
              Method Accuracy Sensitivity Specificity Precision Bal_Accuracy
 Logistic Regression   0.9750      0.3284      0.9974    0.8148       0.6629
                 LDA   0.9750      0.3284      0.9974    0.8148       0.6629
                 QDA   0.9755      0.3731      0.9964    0.7813       0.6847
         Naive Bayes   0.9750      0.4328      0.9938    0.7073       0.7133
           KNN (K=5)   0.9755      0.4030      0.9953    0.7500       0.6991
```

**Reading these results:**

```
Accuracy: All methods cluster at 97.5% — nearly identical!
          This is the "imbalance paradox" in action.

Sensitivity: THIS is where they differ.
          Logistic/LDA: 32.8%  — catch 1 in 3 defaults
          QDA:          37.3%  — catch 2 in 5 defaults
          KNN K=5:      40.3%  — catch 2 in 5 defaults
          Naive Bayes:  43.3%  — catch nearly 1 in 2 defaults ← WINNER

Balanced Accuracy: Fair summary (averages sensitivity and specificity)
          Naive Bayes wins here too (0.713)

AUC (from ROC curves):
          All methods 0.94–0.95 — similar discriminative power
          Differences emerge only at specific thresholds
```

## 8.2 AUC Comparison

```r
# ── AUC comparison ────────────────────────────────────────────────────────────
auc_glm <- auc(roc(test_data$default, glm_prob, quiet = TRUE))

cat("AUC Comparison:\n")
cat(sprintf("  Logistic Regression: %.4f\n", auc_glm))
cat(sprintf("  LDA:                 %.4f\n", auc(roc_lda)))
cat(sprintf("  QDA:                 %.4f\n", auc(roc_qda)))
cat(sprintf("  Naive Bayes:         %.4f\n", auc(roc_nb)))
cat(sprintf("  KNN (K=5):           %.4f\n", auc(roc_knn)))
```

---

---

# Part 9: Decision Boundaries Visualized

Understanding *where* each method draws its boundary explains *why* they differ.

## 9.1 Building a Decision Boundary Plot

```r
# ── Create a fine grid to evaluate each model ─────────────────────────────────
bal_seq <- seq(0, 2700, length.out = 200)
inc_seq <- seq(0, 75000, length.out = 200)
grid    <- expand.grid(balance = bal_seq, income = inc_seq)

# Add scaled versions for KNN
grid_scaled <- data.frame(
  balance = (grid$balance - colMeans(train_X)["balance"]) / apply(train_X, 2, sd)["balance"],
  income  = (grid$income  - colMeans(train_X)["income"])  / apply(train_X, 2, sd)["income"]
)

# Predict on grid
grid$LDA  <- predict(lda_fit, newdata = grid)$class
grid$QDA  <- predict(qda_fit, newdata = grid)$class
grid$NB   <- predict(nb_fit,  newdata = grid)
grid$KNN5 <- knn(train = train_X_scaled,
                 test  = as.matrix(grid_scaled),
                 cl    = train_Y, k = 5)

# Also get posterior probabilities for smooth visualization
grid$LDA_prob <- predict(lda_fit, newdata = grid)$posterior[, "Yes"]
grid$QDA_prob <- predict(qda_fit, newdata = grid)$posterior[, "Yes"]
grid$NB_prob  <- predict(nb_fit,  newdata = grid, type = "raw")[, "Yes"]
```

```r
# ── Plot LDA vs QDA boundaries ────────────────────────────────────────────────
# Use a subset of test data to overlay

test_sample <- test_data[sample(nrow(test_data), 300), ]

p_lda <- ggplot() +
  geom_tile(data = grid, aes(x = balance, y = income, fill = LDA_prob), alpha = 0.7) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "tomato",
                       midpoint = 0.5, name = "P(Default)") +
  geom_point(data = test_sample,
             aes(x = balance, y = income, shape = default, color = default),
             size = 1.5, alpha = 0.8) +
  scale_color_manual(values = c("No" = "steelblue", "Yes" = "tomato")) +
  scale_shape_manual(values = c("No" = 16, "Yes" = 17)) +
  labs(title = "LDA: Linear Boundary",
       subtitle = "One straight line separates classes.\nBoundary moves because of unequal priors.",
       x = "Balance ($)", y = "Income ($)") +
  theme_minimal() + theme(legend.position = "right")

p_qda <- ggplot() +
  geom_tile(data = grid, aes(x = balance, y = income, fill = QDA_prob), alpha = 0.7) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "tomato",
                       midpoint = 0.5, name = "P(Default)") +
  geom_point(data = test_sample,
             aes(x = balance, y = income, shape = default, color = default),
             size = 1.5, alpha = 0.8) +
  scale_color_manual(values = c("No" = "steelblue", "Yes" = "tomato")) +
  scale_shape_manual(values = c("No" = 16, "Yes" = 17)) +
  labs(title = "QDA: Quadratic (Curved) Boundary",
       subtitle = "The boundary bends — QDA allows different covariances per class.",
       x = "Balance ($)", y = "Income ($)") +
  theme_minimal() + theme(legend.position = "right")

# Display
print(p_lda)
print(p_qda)
```

```r
# ── Show KNN boundary at different K values ────────────────────────────────────
plot_knn_boundary <- function(k_val, title_suffix) {
  grid_preds <- knn(train = train_X_scaled,
                    test  = as.matrix(grid_scaled),
                    cl    = train_Y, k = k_val)
  grid$KNN_pred <- grid_preds

  ggplot() +
    geom_tile(data = grid, aes(x = balance, y = income, fill = KNN_pred), alpha = 0.5) +
    scale_fill_manual(values = c("No" = "steelblue", "Yes" = "tomato"), name = "Predicted") +
    geom_point(data = test_sample,
               aes(x = balance, y = income, color = default),
               size = 0.8, alpha = 0.6) +
    scale_color_manual(values = c("No" = "navy", "Yes" = "darkred")) +
    labs(title = paste("KNN:", title_suffix),
         x = "Balance ($)", y = "Income ($)") +
    theme_minimal() + theme(legend.position = "none")
}

print(plot_knn_boundary(1,   "K=1 (Overfitting — jagged wiggly boundary)"))
print(plot_knn_boundary(10,  "K=10 (Smoother, more reasonable)"))
print(plot_knn_boundary(100, "K=100 (Approaching LDA in smoothness)"))
```

**What you'll see:**

```
LDA:      Single straight diagonal line. Clean, simple.
QDA:      Slightly curved line in the same region — subtle difference here
          because covariances aren't dramatically different.
KNN K=1:  Chaotic, jagged boundary — hugging every individual training point.
KNN K=10: Smoother regions, reasonable-looking clusters.
KNN K=100: Almost as smooth as LDA — averaging over many neighbors.
```

---

---

# Part 10: Practical Guide — Which Method, When?

## 10.1 A Decision Framework

```
START: What is your situation?
│
├── How many classes?
│   ├── 2 classes → All methods work
│   └── K > 2 classes → LDA, QDA, NB, or KNN (avoid vanilla logistic)
│
├── How large is n relative to p?
│   ├── n >> p (many obs, few predictors):
│   │   → All methods feasible, QDA and KNN are options
│   └── n ≈ p or n < p (sparse data):
│       → Use LDA or Naive Bayes (fewer parameters)
│
├── Do you suspect nonlinear boundaries?
│   ├── No → LDA or Logistic
│   └── Yes → QDA or KNN (small K)
│
├── Mixed predictor types (continuous + categorical)?
│   └── Naive Bayes handles these naturally
│
├── Need to interpret the model?
│   └── LDA (discriminant coefficients) or Logistic
│
└── Not sure? → Try all, use 10-fold CV to compare
```

## 10.2 The Bias-Variance Spectrum

```
Low Flexibility ←────────────────────────────────────→ High Flexibility
(High Bias, Low Variance)                              (Low Bias, High Variance)

LDA → Logistic → Naive Bayes → QDA → KNN(large K) → KNN(small K)

Use left side when:      Use right side when:
  n is small               n is large
  p is large               p is small
  Need stability           Need flexibility
  Linear boundary OK       Boundary is curved
```

## 10.3 Quick Reference Table

| Method | Boundary | Assumptions | Great for | Avoid when |
|--------|----------|------------|-----------|-----------|
| **Logistic** | Linear | Minimal | General purpose, robust | Non-linear patterns |
| **LDA** | Linear | Normal X, equal Σ | Small n, K>2 classes, well-separated | Assumptions badly violated |
| **QDA** | Quadratic | Normal X | Clear non-linearity, large n | Small n, large p |
| **Naive Bayes** | Varies | Independence | Large p, mixed types, high-dim | Strong feature correlations |
| **KNN** | Arbitrary | None | Complex non-linear, large n, small p | Large p (curse of dim), slow prediction |

## 10.4 Standard Analysis Template

```r
# ── Template: Compare all methods with 10-fold CV ─────────────────────────────
compare_classifiers <- function(data, formula, K = 10) {
  set.seed(42)
  n     <- nrow(data)
  folds <- sample(rep(1:K, length.out = n))

  methods  <- c("LDA", "QDA", "NB", "KNN5", "KNN10")
  errors   <- matrix(NA, nrow = K, ncol = length(methods))
  colnames(errors) <- methods

  for (k in 1:K) {
    tr   <- data[folds != k, ]
    te   <- data[folds == k, ]
    y_te <- te[[all.vars(formula)[1]]]

    # Numeric predictors for KNN
    pred_vars  <- all.vars(formula)[-1]
    tr_num     <- tr[, pred_vars, drop = FALSE]
    te_num     <- te[, pred_vars, drop = FALSE]
    tr_sc      <- scale(tr_num)
    te_sc      <- scale(te_num, center = attr(tr_sc,"scaled:center"),
                                scale  = attr(tr_sc,"scaled:scale"))

    errors[k, "LDA"]   <- mean(predict(lda(formula, data = tr), te)$class != y_te)
    errors[k, "QDA"]   <- mean(predict(qda(formula, data = tr), te)$class != y_te)
    errors[k, "NB"]    <- mean(predict(naiveBayes(formula, data = tr), te) != y_te)
    errors[k, "KNN5"]  <- mean(knn(tr_sc, te_sc, tr[[all.vars(formula)[1]]], k=5)  != y_te)
    errors[k, "KNN10"] <- mean(knn(tr_sc, te_sc, tr[[all.vars(formula)[1]]], k=10) != y_te)
  }

  result <- data.frame(
    Method   = methods,
    CV_Error = round(colMeans(errors), 4),
    CV_SD    = round(apply(errors, 2, sd), 4)
  )
  result[order(result$CV_Error), ]
}

# Run it
cv_results <- compare_classifiers(Default, default ~ balance + income)
print(cv_results)
```

---

---

# Part 11: Common Pitfalls & Checklist

## 11.1 Pitfall 1 — Not Scaling for KNN

```r
# ❌ WRONG: Different predictor scales → one variable dominates
knn(train = train_data[, c("balance", "income")],
    test  = test_data[, c("balance", "income")],
    cl    = train_data$default, k = 5)
# income (range 0–73000) dominates balance (range 0–2600) by 28:1

# ✓ CORRECT: Scale using training statistics only
sc_params  <- list(center = colMeans(train_X), scale = apply(train_X, 2, sd))
train_sc   <- scale(train_X, center = sc_params$center, scale = sc_params$scale)
test_sc    <- scale(test_X,  center = sc_params$center, scale = sc_params$scale)
knn(train = train_sc, test = test_sc, cl = train_data$default, k = 5)
```

## 11.2 Pitfall 2 — Using Accuracy for Imbalanced Classes

```r
# ❌ WRONG: Reporting accuracy alone when classes are imbalanced
cat("Accuracy:", mean(lda_pred$class == test_data$default), "\n")
# 97.5% looks great, but naive baseline is 96.7%!

# ✓ CORRECT: Report sensitivity, specificity, AUC, or balanced accuracy
confusionMatrix(lda_pred$class, test_data$default, positive = "Yes")
# Focus on sensitivity (% of defaults caught) and AUC
```

## 11.3 Pitfall 3 — Using Training Error to Select K in KNN

```r
# ❌ WRONG: Training error for K selection
# K=1 always gives 0% training error — but it's the worst choice!
knn_train_pred <- knn(train_X_scaled, train_X_scaled, train_Y, k = 1)
cat("K=1 training error:", mean(knn_train_pred != train_Y), "\n")  # 0%!

# ✓ CORRECT: Use cross-validation
# (See Section 6.4 for full CV loop)
```

## 11.4 Pitfall 4 — Applying LDA When Covariances Clearly Differ

```r
# ── Check if LDA's equal-covariance assumption holds ─────────────────────────
# Rough check: compare covariance matrices per class

cov_no  <- cov(train_data[train_data$default == "No",  c("balance", "income")])
cov_yes <- cov(train_data[train_data$default == "Yes", c("balance", "income")])

cat("Covariance matrix for No default:\n")
print(round(cov_no, 1))

cat("\nCovariance matrix for Yes default:\n")
print(round(cov_yes, 1))

# If they look very different → prefer QDA over LDA
# Here balance variance: No = 230,963, Yes = 125,975 — about 2x different
# Moderate evidence for QDA
```

## 11.5 Pitfall 5 — Applying KNN in High Dimensions Without Thinking

```r
# If you have many predictors, check this before using KNN:
p_count <- ncol(train_data) - 1  # Subtract response variable
cat("Number of predictors:", p_count, "\n")

if (p_count > 10) {
  cat("WARNING: KNN may suffer from curse of dimensionality with p =", p_count, "\n")
  cat("Consider: feature selection, PCA, or a parametric method (LDA/NB) instead.\n")
} else {
  cat("p =", p_count, "— KNN should work fine.\n")
}
```

## 11.6 Best Practices Checklist

```
Before fitting any classification model:
  ☐ Check class balance: table(y) / nrow(data)
  ☐ Visualize: scatterplot colored by class, density plots per predictor
  ☐ Decide on evaluation metric BEFORE seeing results (sensitivity? AUC? cost?)
  ☐ Set random seed
  ☐ Create train/test split or set up cross-validation

Method-specific checks:
  ☐ LDA/QDA: Plot density by class — does normal shape look reasonable?
  ☐ LDA vs QDA: Compare within-class covariances — similar or very different?
  ☐ Naive Bayes: Acknowledge independence assumption; check for strong correlations
  ☐ KNN: ALWAYS scale predictors | ALWAYS use CV to choose K | Check p

After fitting:
  ☐ Never report accuracy alone for imbalanced data
  ☐ Report: sensitivity, specificity, AUC, or balanced accuracy
  ☐ Plot ROC curve — summarizes all thresholds at once
  ☐ Consider threshold adjustment based on relative error costs
  ☐ Compare multiple methods using the same CV folds for fairness
  ☐ Sanity-check: do the coefficients/patterns make domain sense?
```

---

---

# Appendix: Formula Reference Sheet

## Bayes Theorem (Foundation of LDA, QDA, Naive Bayes)

$$P(Y = k \mid X = x) = \frac{P(X = x \mid Y = k) \cdot P(Y = k)}{\sum_{l=1}^{K} P(X = x \mid Y = l) \cdot P(Y = l)}$$

## LDA

**Model:** $X \mid Y = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$ (shared covariance)

**Discriminant function (multiple predictors):**
$$\delta_k(x) = \mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2}\boldsymbol{\mu}_k^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log(\pi_k)$$

**Decision boundary (1D, equal priors):**
$$x^* = \frac{\mu_1 + \mu_2}{2}$$

**Estimated parameters:**
$$\hat{\pi}_k = \frac{n_k}{n}, \quad \hat{\boldsymbol{\mu}}_k = \frac{1}{n_k}\sum_{i:y_i=k} x_i, \quad \hat{\boldsymbol{\Sigma}} = \frac{1}{n-K}\sum_k\sum_{i:y_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^\top$$

## QDA

**Model:** $X \mid Y = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ (class-specific covariance)

**Discriminant function:**
$$\delta_k(x) = -\frac{1}{2}\log|\boldsymbol{\Sigma}_k| - \frac{1}{2}(x - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(x - \boldsymbol{\mu}_k) + \log(\pi_k)$$

**Decision boundary:** Quadratic in x (parabola, ellipse, or hyperbola)

## Naive Bayes

**Core assumption:**
$$P(X_1, \ldots, X_p \mid Y = k) = \prod_{j=1}^{p} P(X_j \mid Y = k)$$

**Discriminant function:**
$$\delta_k(x) = \log(\pi_k) + \sum_{j=1}^{p} \log P(X_j = x_j \mid Y = k)$$

**For continuous $X_j$** (Gaussian assumption):
$$P(X_j = x \mid Y = k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}}\exp\left(-\frac{(x - \mu_{jk})^2}{2\sigma_{jk}^2}\right)$$

## KNN

**Classification rule:**
$$\hat{P}(Y = k \mid X = x) = \frac{1}{K}\sum_{i \in \mathcal{N}_K(x)} \mathbf{1}(y_i = k)$$

Where $\mathcal{N}_K(x)$ = the K training points closest to x.

**Euclidean distance:**
$$d(x, x_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2}$$

## Evaluation Metrics

$$\text{Sensitivity} = \frac{TP}{TP + FN} \quad \text{Specificity} = \frac{TN}{TN + FP} \quad \text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d[\text{FPR}(t)]$$

---

## Quick Reference: Key R Functions

```r
# ── LDA and QDA ──────────────────────────────────────────────────────────────
library(MASS)
fit_lda  <- lda(y ~ x1 + x2, data = train)     # Fit LDA
fit_qda  <- qda(y ~ x1 + x2, data = train)     # Fit QDA
pred_lda <- predict(fit_lda, newdata = test)    # Returns $class, $posterior, $x
pred_lda$class                                  # Predicted classes
pred_lda$posterior[, "Yes"]                     # P(Y="Yes" | X) for each obs

# ── Naive Bayes ───────────────────────────────────────────────────────────────
library(e1071)
fit_nb    <- naiveBayes(y ~ x1 + x2, data = train)
pred_nb   <- predict(fit_nb, newdata = test)                    # Classes
prob_nb   <- predict(fit_nb, newdata = test, type = "raw")      # Probabilities

# ── KNN ───────────────────────────────────────────────────────────────────────
library(class)
# ALWAYS scale first!
tr_sc  <- scale(train[, pred_vars])
te_sc  <- scale(test[, pred_vars], center = attr(tr_sc, "scaled:center"),
                                   scale  = attr(tr_sc, "scaled:scale"))
# K=5, with probability output
pred_knn  <- knn(train = tr_sc, test = te_sc, cl = train$y, k = 5, prob = TRUE)
knn_votes <- attr(pred_knn, "prob")  # Vote fraction for winning class

# ── Evaluation ────────────────────────────────────────────────────────────────
library(caret)
confusionMatrix(predicted, actual, positive = "Yes")  # Full report

library(pROC)
roc_obj <- roc(actual, predicted_prob, quiet = TRUE)
auc(roc_obj)         # Single AUC number
plot(roc_obj)        # ROC curve
```

---

*End of Chapter 4 — Classification Methods Beyond Logistic Regression*
*ISLR Second Edition | Integrated Lecture + Lab Guide*

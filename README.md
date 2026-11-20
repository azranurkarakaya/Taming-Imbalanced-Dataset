# Taming the Imbalanced Dataset

## About the Project

In this homework, I used Logistic Regression to predict risky credit applicants. I compared different methods to improve the detection of Risk (class 1).

## Dataset

I used the German Credit dataset with 1,000 applicants.

- 0 = Safe (700 applicants)
- 1 = Risk (300 applicants)

The original dataset used 1 = Good and 2 = Bad. I changed these labels to 0 = Safe and 1 = Risk.

I converted categorical features into numbers, scaled numerical features, and checked for missing values.

I used one-hot encoding for categorical features. The code included median imputation for numerical features and most-frequent imputation for categorical features. There were no missing values in the dataset.

I used 80% of the data for training and 20% for testing with `random_state=42`.

Both splits were stratified to keep the same class proportions. I divided the 800 training rows into 600 fitting rows and 200 validation rows.

During validation, preprocessing was fitted only on the 600 fitting rows. After selecting the threshold, I trained the final models on all 800 training rows. All final models shared the same preprocessing fitted on the original training rows.

## Methods

I compared four methods:

1. **Baseline:** Normal Logistic Regression.
2. **Class Weights:** Gives more importance to the Risk class.
3. **Threshold Moving:** Tests different thresholds (0.2, 0.3, 0.4). I selected 0.2 using validation F1-score.
4. **Random Oversampling:** Duplicates Risk examples in the training data to balance the classes.

The baseline used `LogisticRegression(max_iter=1000, random_state=42)`. For class weights, I added `class_weight="balanced"` and kept the other settings the same.

The validation F1-scores were:
- Threshold 0.2: 0.5529
- Threshold 0.3: 0.5205
- Threshold 0.4: 0.4580

I selected 0.2 because it had the highest validation F1-score. A Risk probability of at least 0.2 was classified as Risk.

Random oversampling used `random_state=42` and produced 560 Safe and 560 Risk training examples. It was applied after the training-only preprocessing.

The test data was not used during training or threshold selection.

No convergence warnings occurred, so `max_iter` remained 1000.

## Results

Precision, recall, and F1 refer to class 1 = Risk.

| Method | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.7800 | 0.6667 | 0.5333 | 0.5926 |
| Class Weights | 0.7500 | 0.5581 | 0.8000 | 0.6575 |
| Threshold (0.2) | 0.6550 | 0.4587 | 0.8333 | 0.5917 |
| Random Oversampling | 0.7300 | 0.5349 | 0.7667 | 0.6301 |

## Reflection Questions

**1. Which method had the highest recall?**

Threshold moving (0.2) had the highest recall (0.8333). However, its precision decreased from 0.6667 to 0.4587 compared with the baseline.

It found 50 of the 60 Risk applicants, while the baseline found 32. The lower threshold did not improve F1 in this test.

**2. Why shouldn't we oversample before splitting?**

Because the same examples could appear in both training and test data. This would make the results unreliable.

Oversampling should be done only on the training data.

**3. How would you balance precision and recall as a bank manager?**

I would try to detect risky applicants without rejecting too many safe applicants. In my results, class weights gave a good balance and had the highest F1-score.

Approving a risky applicant can cause financial losses. Rejecting a safe applicant can also lose business. I would compare both costs and choose the operating threshold using validation data. The highest F1-score does not always mean the lowest financial loss.

## How to Run

Install the libraries:

```bash
python -m pip install numpy pandas scikit-learn
```

Run the code:

```bash
python credit_homework.py
```

The code saves the results in:
- comparison_results.csv
- validation_results.csv

## Dataset Source

German Credit Data — UCI Machine Learning Repository

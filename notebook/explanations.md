# Cinnamon Oil Classification — Plain English Explanation

> This document explains every decision made in the notebook in simple language.
> No technical background is needed to understand this.

---

## Table of Contents

1. [What Is This Project Trying to Do?](#1-what-is-this-project-trying-to-do)
2. [What Is the Input Data?](#2-what-is-the-input-data)
3. [Why Is `result` the Target Variable and Not `density`?](#3-why-is-result-the-target-variable-and-not-density)
4. [What Are the 7 Output Classes?](#4-what-are-the-7-output-classes)
5. [Why Do We Split the Data into Train and Validation?](#5-why-do-we-split-the-data-into-train-and-validation)
6. [Why Do We Do EDA (Exploratory Data Analysis)?](#6-why-do-we-do-eda-exploratory-data-analysis)
7. [Why Do We Scale / Preprocess the Numbers?](#7-why-do-we-scale--preprocess-the-numbers)
8. [Why Do We Encode the Labels?](#8-why-do-we-encode-the-labels)
9. [Why Three Tabular Models? (RandomForest, XGBoost, CatBoost)](#9-why-three-tabular-models-randomforest-xgboost-catboost)
10. [Why Do We Use an Image Model?](#10-why-do-we-use-an-image-model)
11. [Why Do We Use a Hybrid Model?](#11-why-do-we-use-a-hybrid-model)
12. [What Is the Confusion Matrix?](#12-what-is-the-confusion-matrix)
13. [What Is Accuracy vs F1 Score?](#13-what-is-accuracy-vs-f1-score)
14. [Why Do We Save the Best Model?](#14-why-do-we-save-the-best-model)
15. [Why Is There a Single-Sample Prediction Helper?](#15-why-is-there-a-single-sample-prediction-helper)
16. [The Full Picture — How Everything Connects](#16-the-full-picture--how-everything-connects)

---

## 1. What Is This Project Trying to Do?

**In plain English:**
We are building a computer program that can look at the measurable properties of a cinnamon oil sample and automatically tell you what type of oil it is — just like a trained quality inspector would, but faster and consistent every single time.

**The real-world problem:**
When cinnamon oil is produced and sold, there are different grades and types:

- High-quality bark oils (`bark_superior`, `bark_special`, `bark_average`, `bark_ordinary`)
- Leaf oils (`leaf_pass`, `leaf_fail`)
- Fake or diluted oils (`adulterated`)

Telling these apart usually requires a human expert or expensive lab work. This project trains a computer to do that job using just two simple measurements — **density** and **pH value** — and optionally a **photo** of the sample.

---

## 2. What Is the Input Data?

The data is stored in a file called `input.csv`. A CSV file is simply a spreadsheet saved as plain text. Each row in this file represents one oil sample that was tested.

Each row has 4 columns:

| Column | What It Contains | Example |
|---|---|---|
| `image_name` | The filename of a photo of the sample | `sample_001.jpg` |
| `density` | How heavy the oil is compared to water | `1.042` |
| `ph_value` | How acidic or alkaline the oil is (0–14 scale) | `5.35` |
| `result` | The correct oil type label for this sample | `bark_superior` |

The dataset contains **700 samples** — 100 samples for each of the 7 oil types. This balance is important because the computer needs to see enough examples of each type to learn them all equally well.

---

## 3. Why Is `result` the Target Variable and Not `density`?

This is one of the most fundamental decisions in the entire project, so it deserves a careful explanation.

**What is a "target variable"?**
The target variable is the answer we want the computer to predict. Everything else is information we give to the computer as clues.

**Think of it like a doctor visit:**
- A patient comes in with symptoms: fever (38.5°C), heart rate (95 bpm), blood pressure (130/85)
- The doctor uses those measurements to reach a **diagnosis** (flu, pneumonia, etc.)
- The *measurements* are the inputs. The *diagnosis* is the target — the answer.

**In our case:**
- The lab technician tests a cinnamon oil sample and measures: density (1.042), pH (5.35)
- We want the system to output: **"This is bark_superior oil"**
- So `density` and `ph_value` are the **inputs (clues)**, and `result` is the **target (answer)**

**Why NOT use `density` as the target?**

If we predicted `density` as the output, the system would answer questions like:
> *"Given this sample, I predict its density is 1.041"*

But we already **measured** the density before we even ran the model! We don't need the computer to tell us something we already know. That would be completely pointless.

What we actually need the computer to tell us is:
> *"Given that the density is 1.041 and pH is 5.35, this oil is **bark_superior**"*

That is useful. That saves lab time. That is why `result` is the target.

**Summary:**

| | `density` as Target | `result` as Target |
|---|---|---|
| What the computer predicts | A number like 1.041 | A label like "bark_superior" |
| Is this useful? | ❌ No — we already measured it | ✅ Yes — this is what we want to know |
| Type of problem | Regression (predicting a number) | Classification (predicting a category) |

---

## 4. What Are the 7 Output Classes?

The model can produce exactly one of these 7 answers for any given sample:

| Output Label | What It Means |
|---|---|
| `bark_superior` | Top-grade cinnamon bark oil — highest quality |
| `bark_special` | Second-grade bark oil — still high quality |
| `bark_average` | Average-grade bark oil — acceptable quality |
| `bark_ordinary` | Lowest-grade bark oil — meets minimum standards |
| `leaf_pass` | Oil extracted from cinnamon leaves — passes quality check |
| `leaf_fail` | Oil extracted from cinnamon leaves — fails quality check |
| `adulterated` | Fake or diluted oil — mixed with other substances |

These 7 categories cover every possible outcome for cinnamon oil quality testing. The model's job is to figure out which one applies to any new sample it sees.

---

## 5. Why Do We Split the Data into Train and Validation?

**The core idea:**
We never want to test the computer on questions it has already seen the answers to. That would be like giving students a test using the exact same questions they studied — it doesn't tell you if they truly understood the subject.

**What we do instead:**
We split our 700 samples into two separate groups:

- **Training set (80% = 560 samples):** The computer studies these samples, sees the correct answers, and learns the patterns.
- **Validation set (20% = 140 samples):** These samples are hidden from the computer during training. Afterwards, we show them to the model as a "surprise test" to see how well it really learned.

**Why does this matter?**
If we trained and tested on the same data, a model could just memorize all the answers without actually understanding anything. By testing on new, unseen samples, we get an honest measure of how the model would perform in the real world.

**What is "stratify"?**
When we split the data, we use something called stratified splitting. This ensures the 80/20 split is fair across all 7 oil types. Without this, you might accidentally end up with a training set that has very few `adulterated` examples, making the model bad at detecting fake oils.

---

## 6. Why Do We Do EDA (Exploratory Data Analysis)?

**EDA means: look at the data before training, to understand what you are working with.**

We produce several charts:

**Chart 1 — Class Distribution Bar Chart**
Shows how many samples exist for each oil type. We want roughly equal counts. If one type had 500 samples and another had only 10, the model would become biased — it would always lean toward predicting the common type.

**Chart 2 — Density Box Plot**
Shows how density values differ across oil types. If `bark_superior` samples consistently have higher density than `adulterated` samples, density is a strong clue for the model to use. This chart helps us visually confirm that.

**Chart 3 — pH Value Box Plot**
Same idea — shows how pH values differ across oil types.

**Chart 4 — Scatter Plot (density vs pH, coloured by type)**
Shows all 700 samples on one graph where the X-axis is density and the Y-axis is pH. Each oil type gets a different colour. If the 7 colour clusters are clearly separate, the model will have an easy job. If they are all mixed together, classification will be harder.

**Chart 5 — Missing Values Check**
Checks if any rows have missing data (blank cells). Missing data can cause errors or mislead the model, so we identify and handle it early.

**Why bother?**
EDA catches problems early. It also builds confidence — if the charts show clear separation between oil types, we know the model has a good chance of success before we even start training.

---

## 7. Why Do We Scale / Preprocess the Numbers?

**The problem with raw numbers:**
Density values are around 1.00–1.05. pH values are around 4.4–5.5. While these ranges seem similar, some machine learning models are sensitive to the scale of numbers. If one feature had values in the thousands and another in the hundredths, the model might incorrectly treat the large-number feature as more important.

**What scaling does:**
Scaling (specifically `StandardScaler`) transforms all numbers so they are on a comparable scale — roughly centred around 0 with a range of about -3 to +3. The actual relationships and patterns in the data stay exactly the same, but the numbers become easier for the model to work with.

**What the Imputer does:**
Before scaling, we also apply a `SimpleImputer`. This fills in any missing values using the median value for that column. It is a safety net — if a sample is missing a density reading, the system does not crash; it uses a sensible substitute.

**Analogy:**
Think of it like converting units. Whether you measure height in centimetres or inches, the person is the same height. Scaling is just making sure all measurements speak the same language.

---

## 8. Why Do We Encode the Labels?

**The problem:**
Computers work with numbers, not words. The model cannot directly process labels like `bark_superior` or `adulterated`. We need to convert them to numbers.

**What Label Encoding does:**
It assigns a number to each class:

| Label | Encoded Number |
|---|---|
| `adulterated` | 0 |
| `bark_average` | 1 |
| `bark_ordinary` | 2 |
| `bark_special` | 3 |
| `bark_superior` | 4 |
| `leaf_fail` | 5 |
| `leaf_pass` | 6 |

The model outputs one of these numbers, and we convert it back to the readable label when showing results to the user.

**Why do we fit the encoder on all 7 classes upfront?**
We define all 7 class names at the start and fit the encoder on them, even before loading the data. This ensures the numbering is always consistent and stable — even if a particular dataset happens to be missing one of the 7 types. Without this, the same label could get different numbers depending on what samples happen to be in the file.

---

## 9. Why Three Tabular Models? (RandomForest, XGBoost, CatBoost)

We train three different models on the numerical data (density + pH). Here is why each one is used and what it does:

### RandomForest

**What it is:**
Imagine asking 300 different experts for their opinion on an oil sample. Each expert looks at slightly different information and uses slightly different rules. The final answer is the majority vote of all 300 experts. That is exactly how a Random Forest works — it builds 300 decision trees and combines their votes.

**Why we use it:**
Random Forest is reliable, robust, and rarely overfits (memorizes training data without learning). It works well even when the features are not perfectly scaled, and it can rank which features are most useful.

**Why `class_weight='balanced'`?**
This tells the model to pay equal attention to all 7 oil types, even if one type is slightly underrepresented. It prevents the model from ignoring rare classes.

### XGBoost

**What it is:**
XGBoost is a more advanced tree-based model. Instead of training 300 trees independently (like Random Forest), it trains trees one after another — each new tree tries to fix the mistakes made by the previous trees. This "boosting" approach often produces higher accuracy.

**Why we use it:**
XGBoost is one of the most powerful and widely used models for tabular data. It tends to achieve very high accuracy, especially when the relationship between features and labels is complex.

**`multi:softprob` / `num_class=7`?**
This tells XGBoost we have 7 categories to predict, and to output a probability for each one. The class with the highest probability wins.

### CatBoost

**What it is:**
CatBoost is another boosting model, developed by Yandex (the Russian search engine company). It is particularly good at handling data with categorical features, but it also performs excellently on purely numerical data.

**Why we use it:**
It is fast, often requires less tuning than XGBoost, and frequently matches or beats XGBoost. Running all three and comparing results gives us the best chance of finding the highest-performing model.

### Why train three models instead of just one?

No single model is always best for every dataset. By training all three and comparing them, we let the data decide which approach works best for *this specific* problem. The winner is automatically selected based on the F1 score (explained in Section 13).

---

## 10. Why Do We Use an Image Model?

**What it is:**
Instead of using the numbers (density and pH), an image model looks at a **photo of the oil sample** and learns to classify it based on visual appearance — colour, clarity, texture, and so on.

**What is EfficientNetB0?**
EfficientNetB0 is a deep learning model originally trained on millions of everyday images (photos of animals, objects, landscapes, etc.). It already knows how to detect colours, edges, textures, and shapes. We take this pre-trained model and fine-tune it to recognise cinnamon oil types. This is called **transfer learning** — we borrow knowledge from another task rather than starting from scratch.

**Why "base.trainable = False"?**
The first layer of EfficientNetB0 represents its existing knowledge about general image features. We freeze these layers (keep them unchanged) while training only the final layers on our specific oil type images. This saves huge amounts of training time and prevents the model from "forgetting" what it already knows.

**Why data augmentation?**
During training, we apply random small changes to each image — slight flips, rotations, zoom, contrast adjustments. The actual oil samples don't change, but the model sees slightly different versions of each photo. This makes the model more robust, so it doesn't get confused if a real-world photo is taken at a slightly different angle or lighting condition.

**When does the image model run?**
Only if at least 20 images are found in the uploaded zip file. If no images are provided, the notebook skips this section and relies entirely on the tabular models — which is perfectly fine. Density alone is already a very strong classifier for cinnamon oil.

---

## 11. Why Do We Use a Hybrid Model?

**The idea:**
The tabular model uses numbers (density + pH). The image model uses photos. Each source of information has strengths and weaknesses. What if we combined them?

**What the Hybrid Model does:**
It runs both branches at the same time:
- One branch processes the **image** through EfficientNetB0 and extracts visual patterns.
- Another branch processes the **numbers** (density, pH) through a small neural network.
- Both outputs are **merged together** into one combined layer, which then makes the final prediction.

**Why is this potentially better?**
Sometimes the colour of the oil is enough to spot an adulterated product. Other times, two oils look the same but have very different densities. By using both visual and numerical information together, the model has more evidence to work with and can make a more confident, accurate decision.

**Analogy:**
Think of a doctor diagnosing a patient using both what they see (the patient looks pale, is shivering) AND what the blood test results say. Using both sources of information leads to a more accurate diagnosis than using either alone.

**When does it run?**
Only if images are available AND numerical features exist. If there are no images, there is nothing to combine, so the hybrid model is skipped.

---

## 12. What Is the Confusion Matrix?

After training each model, we produce a **confusion matrix** — a table that shows exactly where the model is getting things right and where it is making mistakes.

**How to read it:**
- Rows = what the sample actually is
- Columns = what the model predicted

Example:

|  | Predicted: bark_superior | Predicted: adulterated |
|---|---|---|
| **Actual: bark_superior** | 18 ✅ (correct) | 2 ❌ (wrong) |
| **Actual: adulterated** | 0 ✅ | 20 ✅ (correct) |

In this example, the model correctly identified 18 bark_superior samples and 20 adulterated samples, but misclassified 2 bark_superior samples as adulterated.

**Why is this useful?**
Overall accuracy tells you the percentage correct, but it hides details. The confusion matrix tells you *which* specific mistakes the model makes. For example, you might discover the model often confuses `leaf_pass` with `leaf_fail` but never confuses `bark_superior` with `adulterated`. That is actionable insight — you know where to focus improvement efforts.

---

## 13. What Is Accuracy vs F1 Score?

We use two measures to judge how good a model is:

### Accuracy
**What it is:** The percentage of all predictions that were correct.

**Formula:** (Number of correct predictions) ÷ (Total predictions) × 100

**Example:** If the model correctly classifies 130 out of 140 validation samples, accuracy = 92.8%

**Limitation:** Accuracy can be misleading. If 95% of your samples were `bark_superior`, a model that always guesses `bark_superior` would have 95% accuracy — but it would be completely useless for detecting the other 6 types.

### Weighted F1 Score
**What it is:** A more honest measure that considers how well the model performs on *each class separately*, then averages them together weighted by how many samples each class has.

**Why we prefer F1 here:** Our dataset is balanced (100 samples per class), but F1 is still a better indicator of true performance when we care equally about all 7 oil types. A high F1 score means the model is good at identifying all types, not just the most common ones.

**In simple terms:** Accuracy rewards getting lots of easy answers right. F1 rewards getting every category right. We use F1 to pick the best model.

---

## 14. Why Do We Save the Best Model?

After training and comparing all models, we save the winner. Saving means writing the trained model to a file on disk.

**Why?**
Training takes time — minutes or hours. If we don't save the model, all that learning disappears the moment the notebook closes. By saving it, we can:

- Load it later without retraining
- Deploy it in a production application (a web app, a mobile app, a quality control system)
- Share it with colleagues

**What gets saved:**

| File | Purpose |
|---|---|
| `best_tabular_model.pkl` | The winning tabular model (RF / XGBoost / CatBoost) |
| `best_image_model.keras` | The trained image model (if images were used) |
| `best_hybrid_model.keras` | The trained hybrid model (if images were used) |
| `label_encoder.pkl` | The mapping between numbers and oil type names |
| `tabular_preprocessor.pkl` | The scaling rules used on density and pH |
| `model_comparison.csv` | A table comparing all models' scores |
| `model_summary.json` | A summary of all settings and file paths |

Everything is also zipped into one file `final_cinnamon_oil_models.zip` for easy download.

---

## 15. Why Is There a Single-Sample Prediction Helper?

**Cell 36 in the notebook defines a function called `predict_oil_type`.**

This function is the bridge between the trained model and real-world use. Instead of running the full notebook every time you want to classify one new sample, you simply call:

```python
predict_oil_type(density=1.038, ph_value=5.41)
```

And you get back:
```
predicted_label:  bark_special
confidence:       0.89
all_probabilities:
  bark_superior   0.04
  bark_special    0.89
  bark_average    0.05
  bark_ordinary   0.01
  leaf_pass       0.00
  leaf_fail       0.00
  adulterated     0.01
```

**What is "confidence"?**
It tells you how sure the model is. 0.89 means 89% confident it is `bark_special`. If confidence is low (e.g. 0.40), it means the sample's measurements are in an ambiguous range and a human expert should double-check.

**Why include all probabilities?**
Seeing the full breakdown helps you understand the model's reasoning. If it says `bark_special` at 45% and `bark_average` at 40%, you know it is a close call. If it says `bark_superior` at 97%, you can be very confident.

---

## 16. The Full Picture — How Everything Connects

Here is the complete journey of the project from start to finish:

```
Raw Data (input.csv + images)
          │
          ▼
  Load & Validate Data
  (check columns, check classes)
          │
          ▼
  Exploratory Analysis
  (charts, distributions, missing values)
          │
          ▼
  Clean & Preprocess
  (fill missing values, scale numbers,
   encode labels, attach image paths)
          │
          ├─────────────────────────────────┐
          │                                 │
          ▼                                 ▼
  Tabular Models                     Image Model
  (RandomForest, XGBoost,            (EfficientNetB0
   CatBoost use density + pH)         uses photos)
          │                                 │
          └──────────────┬──────────────────┘
                         │
                         ▼
                   Hybrid Model
              (combines numbers + photos)
                         │
                         ▼
              Compare All Models
              (accuracy, F1, confusion matrix)
                         │
                         ▼
              Save Best Model
              (to disk for reuse)
                         │
                         ▼
              Predict New Samples
              → Output: bark_superior /
                bark_special / bark_average /
                bark_ordinary / leaf_pass /
                leaf_fail / adulterated
```

**The end goal:**
A quality control officer at a cinnamon oil facility measures the density and pH of a new batch. They enter those two numbers into a system powered by this model. Within seconds, the system outputs whether the oil is high-grade bark oil, leaf oil, or adulterated — with a confidence percentage. No lengthy lab process, no subjective human judgement, just fast and consistent automated grading.

---

*Document prepared for the Cinnamon Oil Classification Project — Tharindu Wijayarathna*
# 🐧 Palmer Penguins — Data Analysis & ML Pipeline

A complete machine learning pipeline applied to the Palmer Penguins dataset, covering data preprocessing, exploratory data analysis, clustering, and species classification using three different algorithms.

This project was developed as part of the *Fundamentals of Data Processing* course at Riga Technical University.

---

## What This Project Does

Starting from raw penguin measurement data, this notebook walks through a full ML workflow:

1. **Data loading & preprocessing** — type casting, missing value imputation, normalization
2. **Exploratory data analysis** — feature plots, histograms, pairplots, correlation heatmap
3. **K-Means clustering** — finding natural groupings using silhouette coefficient
4. **Hierarchical clustering** — visualized via dendrogram (Ward linkage)
5. **Artificial Neural Network (ANN)** — MLP classifier with two different architectures
6. **k-Nearest Neighbors (kNN)** — classification with k=5, compared against ANN

---

## Dataset

**File:** `Dataset/penguins_size.csv`  
**Source:** Palmer Penguins dataset (Horst, Hill & Gorman, 2020)

| Column | Type | Description |
|---|---|---|
| `species` | categorical | Penguin species: Adelie, Chinstrap, Gentoo |
| `island` | categorical | Island of origin: Torgersen, Biscoe, Dream |
| `culmen_length_mm` | float | Bill length in mm |
| `culmen_depth_mm` | float | Bill depth in mm |
| `flipper_length_mm` | float | Flipper length in mm |
| `body_mass_g` | float | Body mass in grams |
| `sex` | categorical | MALE / FEMALE |

**344 records total.** 2 rows have missing numeric measurements (imputed with column mean); 10 rows have missing `sex` values (encoded as 0).

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

The notebook was originally developed on **Google Colab**. If running locally, remove the Google Drive mount cell and update the file path to point to your local `penguins_size.csv`.

---

## How to Run

**On Google Colab (original setup):**

1. Upload `penguins_size.csv` to your Google Drive under `MyDrive/Colab Notebooks/`
2. Open `PR2_Demo.ipynb` in Colab
3. Run all cells in order

**Locally (Jupyter):**

1. Clone the repo and place `penguins_size.csv` in the same directory as the notebook (or update the path)
2. Remove or comment out the `drive.mount(...)` and `os.chdir(...)` cells
3. Change `pd.read_csv('penguins_size.csv')` to your local path if needed
4. Run all cells

```bash
jupyter notebook Code/PR2_Demo.ipynb
```

---

## Notebook Walkthrough

### 1. Preprocessing
- Categorical columns (`species`, `island`, `sex`) cast to `category` type and label-encoded into `_cat` integer columns
- Missing numeric values filled with column mean
- Invalid `sex` codes (−1) replaced with 0
- Four continuous features min-max normalized to [0, 1]

### 2. Data Visualization
- **Line plot** of normalized features — reveals inverse relationship between culmen depth and the other three features
- **Histograms** — shows class imbalance: one species and one island are under-represented
- **Pairplot** (colored by species) — `culmen_length_mm` vs `culmen_depth_mm` and vs `flipper_length_mm` give the best class separation
- **Correlation heatmap** — `flipper_length_mm` and `body_mass_g` are highly correlated (multicollinearity risk for regression tasks)

### 3. K-Means Clustering
Features used: `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`

Silhouette coefficient is computed for k = 2 to 5 to find the optimal number of clusters. Results are plotted and the best k is used to visualize cluster assignments across feature pairs.

### 4. Hierarchical Clustering
Agglomerative clustering with Ward linkage on the same 3-feature subset, visualized as a full dendrogram.

### 5. Artificial Neural Network
70/30 train-test split (`random_state=42`), target = `species_cat`

| Model | Architecture | Activation | Solver | Max iter |
|---|---|---|---|---|
| Model 1 | 2 layers × 100 neurons | logistic | SGD | 1000 |
| Model 2 | 1 layer × 5 neurons | logistic | SGD | 50 |

Performance evaluated via classification report and confusion matrix.

### 6. k-Nearest Neighbors
`KNeighborsClassifier(n_neighbors=5)` trained on the same 70/30 split. Results compared against ANN using the same metrics.

---

## Project Structure

```
Dataset-Penguins/
├── Code/
│   └── PR2_Demo.ipynb      # Full analysis notebook
└── Dataset/
    └── penguins_size.csv   # Raw dataset (344 rows × 7 columns)
```

"""
Fake News Prediction: Machine Learning + Deep Learning + GNN Pipeline
====================================================================

Dataset expected:
    archive.zip
        News _dataset/Fake.csv
        News _dataset/True.csv

How to run in VS Code terminal:
    python -m venv venv
    .\\venv\\Scripts\\activate       # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    python fake_news_full_pipeline.py

Outputs:
    outputs/results_summary.csv
    outputs/model_comparison_accuracy.png
    outputs/model_comparison_f1.png
    outputs/confusion_matrix_*.png
    outputs/roc_curve_*.png
    outputs/training_plot_*.png
"""

# ============================================================
# a. IMPORT LIBRARIES
# ============================================================

import os
import re
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    SimpleRNN,
    LSTM,
    Bidirectional,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# GNN using PyTorch only, no PyTorch-Geometric required
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Keep None to use full dataset. For quick testing, set e.g. 8000.
MAX_SAMPLES = None

# Machine Learning feature settings
TFIDF_MAX_FEATURES = 12000
K_BEST_FEATURES = 8000

# Deep Learning settings
MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128
DL_EPOCHS = 4
BATCH_SIZE = 128

# GNN settings
# GNN is graph-based and can be heavy on CPU/RAM.
# Default uses a stratified sample. Set GNN_MAX_DOCS = None for full graph if your machine is powerful.
GNN_MAX_DOCS = 5000
GNN_TFIDF_FEATURES = 2500
GNN_NEIGHBORS = 8
GNN_EPOCHS = 80

DATA_ZIP = "archive.zip"
EXTRACT_DIR = "dataset_extracted"
OUTPUT_DIR = "outputs"

Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_name(name: str) -> str:
    """Make model name safe for file names."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())


def tokenize_and_stem(text: str, stemmer: PorterStemmer) -> str:
    """
    Data pre-processing:
        1. Tokenization using regex
        2. Stopword removal
        3. Stemming using PorterStemmer
    """
    text = str(text).lower()

    # Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    # Tokenization: keep only alphabetic words
    tokens = re.findall(r"[a-zA-Z]+", text)

    # Remove stopwords and very short words, then stemming
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in ENGLISH_STOP_WORDS and len(token) > 2
    ]

    return " ".join(processed_tokens)


def evaluate_and_plot(model_name, y_true, y_pred, y_score, results, history=None):
    """
    Evaluation and performance:
        1. Accuracy
        2. Precision
        3. Recall
        4. F1-score
        5. AUC
        6. ROC curve
        7. Confusion matrix
        8. Accuracy/loss plot for deep learning models
    """

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan

    results.append(
        {
            "Model": model_name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc,
        }
    )

    print(f"\n========== {model_name} ==========")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}" if not np.isnan(auc) else "AUC      : N/A")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real News", "Fake News"]
    )
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{safe_name(model_name)}.png", dpi=300)
    plt.close()

    # ROC Curve
    if y_score is not None and not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/roc_curve_{safe_name(model_name)}.png", dpi=300)
        plt.close()

    # Accuracy and Loss plots for Deep Learning models
    if history is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get("accuracy", []), label="Training Accuracy")
        plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Plot - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/accuracy_plot_{safe_name(model_name)}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get("loss", []), label="Training Loss")
        plt.plot(history.history.get("val_loss", []), label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Plot - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/loss_plot_{safe_name(model_name)}.png", dpi=300)
        plt.close()


def get_model_score(model, X):
    """Return score/probability for positive class Fake News = 1."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def make_adaboost_classifier():
    """Compatible with old and new sklearn versions."""
    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    try:
        return AdaBoostClassifier(
            estimator=stump,
            n_estimators=100,
            learning_rate=0.8,
            random_state=RANDOM_STATE,
        )
    except TypeError:
        return AdaBoostClassifier(
            base_estimator=stump,
            n_estimators=100,
            learning_rate=0.8,
            random_state=RANDOM_STATE,
        )


# ============================================================
# b. IMPORT DATASET
# ============================================================

def extract_dataset(zip_path=DATA_ZIP, extract_dir=EXTRACT_DIR):
    if not Path(zip_path).exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Put archive.zip in the same folder as this Python file."
        )

    Path(extract_dir).mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Dataset extracted to: {extract_dir}")


def find_csv_file(root_dir, file_name):
    matches = list(Path(root_dir).rglob(file_name))
    if not matches:
        raise FileNotFoundError(f"{file_name} was not found inside {root_dir}")
    return matches[0]


def load_fake_news_dataset():
    extract_dataset()

    fake_csv = find_csv_file(EXTRACT_DIR, "Fake.csv")
    true_csv = find_csv_file(EXTRACT_DIR, "True.csv")

    fake_df = pd.read_csv(fake_csv)
    true_df = pd.read_csv(true_csv)

    # Label convention:
    # Fake News = 1
    # Real News = 0
    fake_df["label"] = 1
    true_df["label"] = 0

    data = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

    print("\nDataset loaded successfully")
    print("Fake.csv shape:", fake_df.shape)
    print("True.csv shape:", true_df.shape)
    print("Combined shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    print("\nLabel distribution:")
    print(data["label"].value_counts())

    return data


# ============================================================
# c. DATA PRE-PROCESSING
#       1. Tokenization
#       2. Stemming
# ============================================================

def preprocess_dataset(data):
    data = data.copy()

    # Combine title and article text for classification
    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")
    data["content"] = data["title"] + " " + data["text"]

    # Remove duplicates and empty content
    data = data.drop_duplicates(subset=["content"]).reset_index(drop=True)
    data = data[data["content"].str.strip() != ""].reset_index(drop=True)

    # Optional sampling for faster testing
    if MAX_SAMPLES is not None and len(data) > MAX_SAMPLES:
        data, _ = train_test_split(
            data,
            train_size=MAX_SAMPLES,
            stratify=data["label"],
            random_state=RANDOM_STATE,
        )
        data = data.reset_index(drop=True)

    stemmer = PorterStemmer()

    print("\nPre-processing started: tokenization + stemming...")
    data["clean_text"] = data["content"].apply(lambda x: tokenize_and_stem(x, stemmer))
    data = data[data["clean_text"].str.strip() != ""].reset_index(drop=True)

    print("Pre-processing completed")
    print("Final dataset shape:", data.shape)
    print("\nSample cleaned text:")
    print(data[["clean_text", "label"]].head())

    return data


# ============================================================
# d. DATA SPLITTING
#       1. Training Set
#       2. Validation Set
#       3. Testing Set
# ============================================================

def split_dataset(data):
    train_df, temp_df = train_test_split(
        data,
        test_size=0.30,
        stratify=data["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    print("\nData Splitting:")
    print("Training Set  :", train_df.shape)
    print("Validation Set:", val_df.shape)
    print("Testing Set   :", test_df.shape)

    X_train = train_df["clean_text"].tolist()
    y_train = train_df["label"].values

    X_val = val_df["clean_text"].tolist()
    y_val = val_df["label"].values

    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].values

    return train_df, val_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# e. FEATURE ENGINEERING
#       1. Feature Extraction
#       2. Feature Selection
# ============================================================

def create_ml_features(X_train, X_val, X_test, y_train):
    print("\nFeature Extraction: TF-IDF started...")

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    print("TF-IDF feature shape:", X_train_tfidf.shape)

    print("\nFeature Selection: Chi-Square SelectKBest started...")

    k = min(K_BEST_FEATURES, X_train_tfidf.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)

    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_val_selected = selector.transform(X_val_tfidf)
    X_test_selected = selector.transform(X_test_tfidf)

    print("Selected feature shape:", X_train_selected.shape)

    return X_train_selected, X_val_selected, X_test_selected, tfidf, selector


# ============================================================
# f. MODEL TRAINING
#       1. MACHINE LEARNING MODELS
# ============================================================

def train_machine_learning_models(X_train_ml, X_test_ml, y_train, y_test, results):
    ml_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "SVM": LinearSVC(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Boosting": make_adaboost_classifier(),
    }

    trained_models = {}

    print("\n================ MACHINE LEARNING MODEL TRAINING ================")

    for model_name, model in ml_models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_ml, y_train)

        y_pred = model.predict(X_test_ml)
        y_score = get_model_score(model, X_test_ml)

        evaluate_and_plot(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            results=results,
        )

        trained_models[model_name] = model

    return trained_models


# ============================================================
# f. MODEL TRAINING
#       2. DEEP LEARNING MODELS
#           {Prediction Model}
# ============================================================

def prepare_deep_learning_data(X_train, X_val, X_test):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    X_val_pad = pad_sequences(
        X_val_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    return tokenizer, X_train_pad, X_val_pad, X_test_pad


def build_cnn_model():
    model = Sequential(
        [
            Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
            Conv1D(filters=128, kernel_size=5, activation="relu"),
            GlobalMaxPooling1D(),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_rnn_model():
    model = Sequential(
        [
            Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
            SimpleRNN(64),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_bilstm_model():
    model = Sequential(
        [
            Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
            Bidirectional(LSTM(64)),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_deep_learning_models(
    X_train_pad,
    X_val_pad,
    X_test_pad,
    y_train,
    y_val,
    y_test,
    results,
):
    dl_models = {
        "CNN": build_cnn_model(),
        "RNN": build_rnn_model(),
        "BI-LSTM": build_bilstm_model(),
    }

    trained_dl_models = {}

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    print("\n================ DEEP LEARNING MODEL TRAINING ================")

    for model_name, model in dl_models.items():
        print(f"\nTraining {model_name}...")
        model.summary()

        history = model.fit(
            X_train_pad,
            y_train,
            validation_data=(X_val_pad, y_val),
            epochs=DL_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1,
        )

        y_score = model.predict(X_test_pad, batch_size=BATCH_SIZE).ravel()
        y_pred = (y_score >= 0.5).astype(int)

        evaluate_and_plot(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            results=results,
            history=history,
        )

        trained_dl_models[model_name] = model

    return trained_dl_models


# ============================================================
# f. MODEL TRAINING
#       3. GRAPH NEURAL NETWORK MODEL
# ============================================================

def normalize_sparse_adjacency(adj):
    """Symmetric adjacency normalization: D^-1/2 A D^-1/2"""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def scipy_sparse_to_torch_sparse(matrix):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack((matrix.row, matrix.col)))
    values = torch.FloatTensor(matrix.data)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        return self.linear(x)


class GCNTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        return x


def build_document_similarity_graph(tfidf_features, k=GNN_NEIGHBORS):
    """
    Build a document graph:
        Nodes = news articles
        Edges = cosine-similar nearest neighbour documents
    """
    print("\nBuilding GNN document similarity graph...")

    nn_model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn_model.fit(tfidf_features)

    distances, indices = nn_model.kneighbors(tfidf_features)

    rows = []
    cols = []
    weights = []

    n_docs = tfidf_features.shape[0]

    for doc_id in range(n_docs):
        for neighbor_position in range(1, k + 1):  # skip itself at position 0
            neighbor_id = indices[doc_id][neighbor_position]
            similarity = 1.0 - distances[doc_id][neighbor_position]
            if similarity > 0:
                rows.append(doc_id)
                cols.append(neighbor_id)
                weights.append(similarity)

    # Create adjacency matrix
    adj = sparse.coo_matrix(
        (weights, (rows, cols)),
        shape=(n_docs, n_docs),
        dtype=np.float32,
    )

    # Make undirected graph and add self-loops
    adj = adj.maximum(adj.T)
    adj = adj + sparse.eye(n_docs, dtype=np.float32)

    adj_norm = normalize_sparse_adjacency(adj)

    print("GNN graph nodes:", n_docs)
    print("GNN graph edges:", adj_norm.nnz)

    return adj_norm


def train_gnn_model(data, results):
    print("\n================ GRAPH NEURAL NETWORK MODEL TRAINING ================")

    gnn_data = data.copy()

    if GNN_MAX_DOCS is not None and len(gnn_data) > GNN_MAX_DOCS:
        gnn_data, _ = train_test_split(
            gnn_data,
            train_size=GNN_MAX_DOCS,
            stratify=gnn_data["label"],
            random_state=RANDOM_STATE,
        )
        gnn_data = gnn_data.reset_index(drop=True)
        print(
            f"GNN is using a stratified sample of {GNN_MAX_DOCS} documents "
            f"to keep training practical on normal laptops."
        )

    train_gnn, temp_gnn = train_test_split(
        gnn_data,
        test_size=0.30,
        stratify=gnn_data["label"],
        random_state=RANDOM_STATE,
    )

    val_gnn, test_gnn = train_test_split(
        temp_gnn,
        test_size=0.50,
        stratify=temp_gnn["label"],
        random_state=RANDOM_STATE,
    )

    train_idx = train_gnn.index.values
    val_idx = val_gnn.index.values
    test_idx = test_gnn.index.values

    # TF-IDF features for graph nodes
    vectorizer = TfidfVectorizer(
        max_features=GNN_TFIDF_FEATURES,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_all = vectorizer.fit_transform(gnn_data["clean_text"].tolist())
    y_all = gnn_data["label"].values

    adj = build_document_similarity_graph(X_all, k=GNN_NEIGHBORS)

    # Convert to torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GNN device:", device)

    features = torch.FloatTensor(X_all.toarray()).to(device)
    labels = torch.LongTensor(y_all).to(device)
    adj_torch = scipy_sparse_to_torch_sparse(adj).to(device)

    train_mask = torch.zeros(len(gnn_data), dtype=torch.bool).to(device)
    val_mask = torch.zeros(len(gnn_data), dtype=torch.bool).to(device)
    test_mask = torch.zeros(len(gnn_data), dtype=torch.bool).to(device)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    model = GCNTextClassifier(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(1, GNN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(features, adj_torch)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj_torch)

            train_pred = logits[train_mask].argmax(dim=1)
            val_pred = logits[val_mask].argmax(dim=1)

            train_acc = (train_pred == labels[train_mask]).float().mean().item()
            val_acc = (val_pred == labels[val_mask]).float().mean().item()
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask]).item()

            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping for GNN")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(features, adj_torch)
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

    y_true = labels[test_mask].detach().cpu().numpy()
    y_pred = predictions[test_mask].detach().cpu().numpy()
    y_score = probabilities[test_mask].detach().cpu().numpy()

    evaluate_and_plot(
        model_name="GNN",
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        results=results,
    )

    # GNN accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_history, label="Training Accuracy")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Plot - GNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_plot_gnn.png", dpi=300)
    plt.close()

    return model


# ============================================================
# h. COMPARE ALL ALGORITHMS USING BAR GRAPH
# i. CONFUSION MATRIX + ACCURACY PLOTS
# ============================================================

def compare_models(results):
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1-Score", ascending=False)

    print("\n================ FINAL MODEL COMPARISON ================")
    print(results_df)

    results_df.to_csv(f"{OUTPUT_DIR}/results_summary.csv", index=False)

    # Accuracy bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison using Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_accuracy.png", dpi=300)
    plt.close()

    # F1-score bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(results_df["Model"], results_df["F1-Score"])
    plt.xlabel("Models")
    plt.ylabel("F1-Score")
    plt.title("Model Comparison using F1-Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_f1.png", dpi=300)
    plt.close()

    # Multi-metric grouped bar graph
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    plot_df = results_df.set_index("Model")[metric_cols]

    ax = plot_df.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Comparison of All Algorithms")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_all_metrics.png", dpi=300)
    plt.close()

    print(f"\nAll output files are saved inside: {OUTPUT_DIR}/")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    # For reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # b. import dataset
    data = load_fake_news_dataset()

    # c. data pre-processing
    data = preprocess_dataset(data)

    # d. data splitting
    (
        train_df,
        val_df,
        test_df,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(data)

    results = []

    # e. feature engineering for ML
    X_train_ml, X_val_ml, X_test_ml, tfidf, selector = create_ml_features(
        X_train,
        X_val,
        X_test,
        y_train,
    )

    # f1. machine learning model training
    trained_ml_models = train_machine_learning_models(
        X_train_ml,
        X_test_ml,
        y_train,
        y_test,
        results,
    )

    # f2. deep learning model training
    tokenizer, X_train_pad, X_val_pad, X_test_pad = prepare_deep_learning_data(
        X_train,
        X_val,
        X_test,
    )

    trained_dl_models = train_deep_learning_models(
        X_train_pad,
        X_val_pad,
        X_test_pad,
        y_train,
        y_val,
        y_test,
        results,
    )

    # f3. graph neural network model training
    gnn_model = train_gnn_model(data, results)

    # h/i. comparison graphs, CSV summary, confusion matrices, accuracy plots
    compare_models(results)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

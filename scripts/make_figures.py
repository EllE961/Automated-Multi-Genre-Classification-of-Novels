"""Re-run the BasicModels.ipynb pipeline and rebuild the README figures.

Run from the repo root (needs collected_books.json, pulled with git lfs):
    python scripts/make_figures.py
The first run preprocesses 762 novels (70M words) on 6 processes and caches the result in
.cache/processed_text.json; the soft-voting ensemble is the other slow step. About 25 minutes the first
time and 18 after that on my machine. Writes PNGs to docs/assets/.
"""
import json
import os
import re
import string
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
CACHE = os.path.join(ROOT, ".cache", "processed_text.json")

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


# ---------------------------------------------------------------- preprocessing, as in BasicModels.ipynb
_STOP = None
_LEMMA = None


def _init_worker():
    global _STOP, _LEMMA
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for res, name in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
                      ("corpora/omw-1.4", "omw-1.4"), ("tokenizers/punkt", "punkt")]:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(name, quiet=True)
    with open(os.path.join(ROOT, "stopwords.txt"), encoding="utf-8") as f:
        user = [line.strip().strip(string.punctuation).lower() for line in f if line.strip()]
    _STOP = set(stopwords.words("english")).union(user)
    _LEMMA = WordNetLemmatizer()


def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    tokens = [w for w in word_tokenize(text) if w not in _STOP and len(w) > 2]
    return " ".join(_LEMMA.lemmatize(w) for w in tokens)


def load_processed(df):
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(df):
            print("using cached preprocessing")
            return [cached[str(i)] for i in df["id"]]
    print("preprocessing 762 novels on 6 processes (a few minutes)...")
    with Pool(6, initializer=_init_worker) as pool:
        out = pool.map(preprocess_text, df["text"].tolist(), chunksize=4)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump({str(i): t for i, t in zip(df["id"], out)}, f)
    return out


# ---------------------------------------------------------------- plotting helpers
def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def caption(fig, text):
    fig.text(0.012, -0.015, text, color=INK3, fontsize=8.5, ha="left", va="top")


def heat(ax, m, labels_x, labels_y, fmt, vmax=None):
    ax.imshow(m, cmap=HEAT, vmin=0, vmax=vmax if vmax is not None else m.max(), aspect="auto")
    ax.grid(False)
    top = vmax if vmax is not None else m.max()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if v == 0:
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.6,
                    color=BG if v > top * 0.55 else INK2)
    ax.set_xticks(range(len(labels_x)), labels_x, rotation=40, ha="right")
    ax.set_yticks(range(len(labels_y)), labels_y)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    from xgboost import XGBClassifier

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(ROOT, "collected_books.json"), encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["processed_text"] = load_processed(df)
    # keep what the checks below need, then drop the raw text (about 430 MB) before vectorizing
    df["words"] = df["text"].str.split().str.len()
    df["audio"] = df["text"].str[:300].str.contains("audio", case=False)
    df["binary"] = df["text"].str.contains("IHDR", regex=False)
    df.drop(columns="text", inplace=True)

    # identical to the notebook: TF-IDF and chi2 selection are fitted on all 762 books, then split
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(df["processed_text"])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["genres"])
    genres = list(mlb.classes_)
    selector = SelectKBest(score_func=chi2, k=3000)
    X_selected = selector.fit_transform(X_tfidf, y)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(msss.split(X_selected, y))
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"books {len(df)}  train {len(train_index)}  test {len(test_index)}  test labels {y_test.sum()}")

    models = {
        "Logistic Regression": OneVsRestClassifier(LogisticRegression(
            penalty="l2", C=10, max_iter=1000, solver="lbfgs", class_weight="balanced")),
        "MLP": OneVsRestClassifier(MLPClassifier(
            hidden_layer_sizes=(200,), activation="tanh", alpha=0.001, learning_rate_init=0.01,
            max_iter=500, random_state=42, early_stopping=True, tol=1e-4)),
        "Random Forest": OneVsRestClassifier(RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10, random_state=42, class_weight="balanced")),
        "XGBoost": OneVsRestClassifier(XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss", learning_rate=0.1, max_depth=6, n_estimators=100,
            random_state=42, n_jobs=2, tree_method="hist", verbosity=0)),
    }
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
    voting = OneVsRestClassifier(VotingClassifier(
        estimators=[(n.lower().replace(" ", "_"), m.estimator) for n, m in models.items()], voting="soft", n_jobs=-1))
    voting.fit(X_train, y_train)
    preds["Soft-voting ensemble"] = voting.predict(X_test)

    rows = {}
    for name, p in preds.items():
        rows[name] = dict(
            macro_f1=f1_score(y_test, p, average="macro", zero_division=0),
            micro_f1=f1_score(y_test, p, average="micro", zero_division=0),
            samples_f1=f1_score(y_test, p, average="samples", zero_division=0),
            exact=accuracy_score(y_test, p), hamming=hamming_loss(y_test, p))
    print(f"{'model':22s} macroF1 microF1 samplesF1 exact  hamming")
    for name, r in rows.items():
        print(f"{name:22s} {r['macro_f1']:.4f}  {r['micro_f1']:.4f}  {r['samples_f1']:.4f}   "
              f"{r['exact']:.4f} {r['hamming']:.4f}")

    best = "Logistic Regression"
    p_best = preds[best]
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, p_best, zero_division=0)
    for g, a, b, c, s in zip(genres, prec, rec, f1, sup):
        print(f"  {g:16s} P {a:.2f}  R {b:.2f}  F1 {c:.2f}  n {s}")
    print(f"  labels per test book: true {y_test.sum(1).mean():.2f}  predicted {p_best.sum(1).mean():.2f}  "
          f"books with no predicted genre {(p_best.sum(1) == 0).sum()}")

    # duplicate editions: the collector keys on Gutenberg ID, so popular titles appear more than once
    titles = df["title"].str.strip().str.lower()
    shared = titles.duplicated(keep=False)
    twin = titles.iloc[test_index].isin(set(titles.iloc[train_index])).values
    print(f"books sharing an exact title with another book: {shared.sum()} ({titles[shared].nunique()} titles); "
          f"test books with a same-title copy in train: {twin.sum()}")
    print(f"LR macro F1 on the {(~twin).sum()} test books without a copy in train: "
          f"{f1_score(y_test[~twin], p_best[~twin], average='macro', zero_division=0):.4f}")

    # the collector caps each genre at 100 books: where in the crawl order did each genre fill up?
    filled, seen = {}, {}
    for pos, labels in enumerate(df["genres"], start=1):
        for g in labels:
            seen[g] = seen.get(g, 0) + 1
            if seen[g] == 100:
                filled[g] = pos
    print("genre full after N books collected:", ", ".join(f"{g} {n}" for g, n in sorted(filled.items(), key=lambda x: x[1])))

    # documents that are not book text: audiobook index pages and binary files
    words = df["words"]
    short = (words < 1000).values
    audio = df["audio"].values
    binary = df["binary"].values & short
    print(f"documents under 1,000 words: {short.sum()}  (audiobook pages: {(audio & short).sum()}, "
          f"PNG data instead of text: {binary.sum()})")
    real_test = ~short[test_index]
    print(f"LR macro F1 on the {real_test.sum()} test books with at least 1,000 words: "
          f"{f1_score(y_test[real_test], p_best[real_test], average='macro', zero_division=0):.4f}")

    # leakage check: fit TF-IDF and chi2 on the training books only
    vec_tr = TfidfVectorizer(max_features=5000)
    A_tr = vec_tr.fit_transform(df["processed_text"].iloc[train_index])
    A_te = vec_tr.transform(df["processed_text"].iloc[test_index])
    sel_tr = SelectKBest(score_func=chi2, k=3000).fit(A_tr, y_train)
    lr_clean = OneVsRestClassifier(LogisticRegression(
        penalty="l2", C=10, max_iter=1000, solver="lbfgs", class_weight="balanced"))
    lr_clean.fit(sel_tr.transform(A_tr), y_train)
    p_clean = lr_clean.predict(sel_tr.transform(A_te))
    clean_macro = f1_score(y_test, p_clean, average="macro", zero_division=0)
    clean_micro = f1_score(y_test, p_clean, average="micro", zero_division=0)
    print(f"LR with TF-IDF + chi2 fitted on train only: macro F1 {clean_macro:.4f}  micro F1 {clean_micro:.4f}")

    # ------------------------------------------------------------ hero: per-genre F1 + genre confusion
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    order = np.argsort(f1)
    yy = np.arange(len(genres))
    ax.barh(yy, f1[order], color=EMBER, height=0.66)
    for yv, v in zip(yy, f1[order]):
        ax.text(v + 0.012, yv, f"{v:.2f}", va="center", color=INK2, fontsize=9,
                bbox=dict(facecolor=PANEL, edgecolor="none", pad=0.6))
    ax.set_yticks(yy, [genres[i] for i in order])
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    macro = rows[best]["macro_f1"]
    ax.axvline(macro, color=INK3, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.text(macro + 0.01, len(genres) - 0.45, f"macro F1 {macro:.2f}", color=INK3, fontsize=8.5)
    ax.set_xlabel("F1 on 152 held-out books")
    ax.set_title("Per-genre F1, one-vs-rest logistic regression")

    ax = fig.add_subplot(gs[1])
    conf = np.zeros((len(genres), len(genres)))
    for i in range(len(genres)):
        has = y_test[:, i] == 1
        conf[i] = p_best[has].mean(axis=0)
    heat(ax, conf, genres, genres, lambda v: f"{v:.2f}", vmax=1.0)
    ax.set_xlabel("genre the model assigned")
    ax.set_ylabel("true genre")
    ax.set_title("Which genres get tagged, by true genre")
    fig.text(0.012, -0.09, "Right: share of test books with the row genre that the model tagged with the column "
             "genre. Books can carry several genres, so rows don't sum to 1; the diagonal is recall.",
             color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "hero.png")

    # ------------------------------------------------------------ data: labels per book, co-occurrence, length
    words = df["words"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [0.8, 1.25, 1.1], "wspace": 0.38})
    ax = axes[0]
    k = pd.Series(y.sum(1)).value_counts().sort_index()
    bars = ax.bar(k.index.astype(str), k.values, color=TEMPER, width=0.6)
    for b, v in zip(bars, k.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(0, k.max() * 1.15)
    ax.set_xlabel("genres per book")
    ax.set_ylabel("books")
    ax.set_title("762 books, 1,000 genre tags")
    ax = axes[1]
    co = (y.T @ y).astype(float)
    np.fill_diagonal(co, 0)
    heat(ax, co, genres, genres, lambda v: f"{int(v)}")
    ax.set_title("Genre pairs on the same book")
    ax = axes[2]
    bins = np.logspace(np.log10(max(words.min(), 10)), np.log10(words.max()), 40)
    ax.hist(words, bins=bins, color=EMBER, alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(words.median(), color=INK3, lw=1, ls=(0, (3, 3)))
    ax.text(words.median() / 1.15, ax.get_ylim()[1] * 0.92, f"median {int(words.median()):,} words",
            color=INK3, fontsize=8.5, ha="right")
    ax.annotate(f"{(words < 1000).sum()} 'books' under\n1,000 words", (600, 30), xytext=(40, 55), color=INK2,
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.set_xlabel("words per book (log scale)")
    ax.set_ylabel("books")
    ax.set_title("Book length")
    save(fig, "data.png")

    # ------------------------------------------------------------ models
    fig, ax = plt.subplots(figsize=(11, 4.6))
    names = list(rows)
    x = np.arange(len(names))
    w = 0.26
    for k_, (key, label, col) in enumerate([("macro_f1", "macro F1", EMBER), ("micro_f1", "micro F1", TEMPER),
                                            ("exact", "exact-set accuracy", PURPLE)]):
        vals = [rows[n][key] for n in names]
        bars = ax.bar(x + (k_ - 1) * w, vals, w * 0.92, color=col, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("score on 152 held-out books")
    ax.set_title("Five models on the same 3,000 chi-squared-selected TF-IDF features")
    ax.legend(ncol=3, loc="upper right")
    save(fig, "models.png")


if __name__ == "__main__":
    sys.exit(main())

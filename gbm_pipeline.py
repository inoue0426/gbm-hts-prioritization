"""Utility functions for GBM HTS prioritization notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
FP_BITS = 1024
NGRAM_RANGE = (2, 5)
MIN_DF = 2
CAT_COLS = ["Cell_line", "library"]


def normalize_name(value: object) -> Optional[str]:
    """Normalize chemical name strings for CTD mapping.

    Args:
        value: Raw value to normalize.

    Returns:
        Normalized lowercase string, or None if missing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    return text or None


def define_hit_label(df: pd.DataFrame) -> pd.Series:
    """Define hit label using HTS criteria.

    Args:
        df: HTS summary dataframe with Max Resp, AC50 (uM), CC-v2 columns.

    Returns:
        Binary label Series where 1 indicates a hit.
    """
    max_resp = pd.to_numeric(df["Max Resp"], errors="coerce")
    ac50 = pd.to_numeric(df["AC50 (uM)"], errors="coerce")
    cc_v2 = df["CC-v2"].astype("string")

    cc_bad = cc_v2.isin(["+/-3", "+/-5", "3", "5", "+3", "-3", "+5", "-5"])

    is_hit = (max_resp < -70) & (ac50 < 15) & (~cc_bad)
    return is_hit.astype(int)


def build_structure_matrix(
    smiles: Sequence[str],
    vectorizer: Optional[TfidfVectorizer] = None,
    use_rdkit: Optional[bool] = None,
) -> Tuple[csr_matrix, Optional[TfidfVectorizer], List[str], str]:
    """Build structure features using RDKit Morgan or SMILES n-grams.

    Args:
        smiles: Iterable of SMILES strings.
        vectorizer: Optional fitted TfidfVectorizer for SMILES n-grams.
        use_rdkit: Force RDKit usage if True, disable if False, autodetect if None.

    Returns:
        Sparse feature matrix, fitted vectorizer (if used), feature names, and method string.
    """
    if use_rdkit is None:
        try:
            from rdkit import Chem  # noqa: F401
            use_rdkit = True
        except Exception:
            use_rdkit = False

    smiles_series = pd.Series(smiles).fillna("")

    if use_rdkit:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")

        fps = []
        for smi in smiles_series.astype(str):
            arr = np.zeros((FP_BITS,), dtype=np.uint8)
            if smi.strip():
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_BITS)
                    DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        X = csr_matrix(np.vstack(fps))
        feature_names = [f"fp_{i}" for i in range(FP_BITS)]
        return X, None, feature_names, "rdkit_morgan"

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
        )
        X = vectorizer.fit_transform(smiles_series.astype(str))
    else:
        X = vectorizer.transform(smiles_series.astype(str))

    feature_names = list(vectorizer.get_feature_names_out())
    return X.tocsr(), vectorizer, feature_names, "smiles_ngrams"


def build_categorical_matrix(
    df: pd.DataFrame,
    ohe: Optional[OneHotEncoder] = None,
    fit: bool = False,
) -> Tuple[csr_matrix, OneHotEncoder, List[str]]:
    """Build categorical one-hot features for selected columns.

    Args:
        df: Input dataframe.
        ohe: Existing encoder to reuse.
        fit: Whether to fit the encoder.

    Returns:
        Sparse matrix, fitted encoder, feature names.
    """
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    if not cat_cols:
        empty = csr_matrix((len(df), 0), dtype=np.float32)
        return empty, ohe or OneHotEncoder(), []

    cat_df = df[cat_cols].astype("string").fillna("unknown")
    if ohe is None:
        try:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32,
            )
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    X = ohe.fit_transform(cat_df) if fit else ohe.transform(cat_df)
    feature_names = list(ohe.get_feature_names_out(cat_cols))
    return X.tocsr(), ohe, feature_names


def load_ctd_mapping(
    path: str,
    feature_col: str,
    chemical_col: str = "ChemicalName",
) -> Dict[str, List[str]]:
    """Load CTD mapping and return chemical -> list of features.

    Args:
        path: TSV file path.
        feature_col: Column name for the feature (e.g., PathwayName, GeneSymbol).
        chemical_col: Column name for chemical names.

    Returns:
        Dict mapping normalized chemical names to sorted unique feature list.
    """
    df = pd.read_table(path)
    df = df.dropna(subset=[chemical_col, feature_col])
    df[chemical_col] = df[chemical_col].map(normalize_name)
    df[feature_col] = df[feature_col].astype("string").str.strip()

    mapping: Dict[str, List[str]] = {}
    for name, group in df.groupby(chemical_col):
        if name is None:
            continue
        features = sorted({f for f in group[feature_col].tolist() if f})
        if features:
            mapping[name] = features
    return mapping


def build_multihot_matrix(
    names: Sequence[Optional[str]],
    mapping: Dict[str, List[str]],
    feature_prefix: str,
    feature_subset: Optional[Iterable[str]] = None,
) -> Tuple[csr_matrix, List[str]]:
    """Build a sparse multi-hot matrix from mapping.

    Args:
        names: Chemical names aligned to samples.
        mapping: Chemical -> list of features.
        feature_prefix: Prefix to attach to feature names.
        feature_subset: Optional subset of features to restrict to.

    Returns:
        Sparse matrix and feature names.
    """
    if feature_subset is None:
        all_features = sorted({f for features in mapping.values() for f in features})
    else:
        all_features = sorted(set(feature_subset))

    feature_to_idx = {f: i for i, f in enumerate(all_features)}
    rows: List[int] = []
    cols: List[int] = []

    for i, name in enumerate(names):
        if not name:
            continue
        for feat in mapping.get(name, []):
            idx = feature_to_idx.get(feat)
            if idx is None:
                continue
            rows.append(i)
            cols.append(idx)

    data = np.ones(len(rows), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(len(names), len(all_features)))
    feature_names = [f"{feature_prefix}::{f}" for f in all_features]
    return mat, feature_names


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute precision@k for binary labels."""
    if k <= 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return float(labels[order].sum() / max(len(order), 1))


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute recall@k for binary labels."""
    total_pos = labels.sum()
    if total_pos == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return float(labels[order].sum() / total_pos)


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute NDCG@k for binary relevance."""
    if k <= 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(labels)[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute mean average precision@k for a single query."""
    order = np.argsort(scores)[::-1][:k]
    hits = labels[order]
    if hits.sum() == 0:
        return 0.0
    precisions = []
    hit_count = 0
    for idx, rel in enumerate(hits, start=1):
        if rel:
            hit_count += 1
            precisions.append(hit_count / idx)
    return float(np.mean(precisions)) if precisions else 0.0

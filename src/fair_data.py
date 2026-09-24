"""
Wrapper over the FLAI library (González-Sendino et al., 2024) to:

  1. Learn the causal graph from the original data.
  2. Mitigate relationships and conditional probabilities (Fair Causal Model).
  3. Generate Fair Data via forward sampling.
  4. Generate data with simple resampling (baseline Liu et al., 2024).

FLAI reference: https://github.com/rugonzs/FLAI
"""

import random

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

try:
    from FLAI import data as flai_data
    from FLAI import causal_graph
    import bnlearn as bn
    print("[fair_data] Backend: FLAI")
except ImportError:
    flai_data = causal_graph = bn = None
    print("[fair_data] WARNING: flai-causal or bnlearn not installed. Run: pip install flai-causal bnlearn")

# Maximum unique values per column allowed for FLAI.
# Columns with more values are automatically discretized into MAX_CARD bins.
MAX_CARD = 10

# Share of the column a single value must reach to get a bin of its own.
DOMINANT_VALUE_SHARE = 0.2


def _bin_continuous(col: pd.Series, n_bins: int) -> pd.Series:
    """Assign a bin code to every observation of a continuous column.

    Values concentrating at least ``DOMINANT_VALUE_SHARE`` of the column get a
    bin of their own, and the remaining observations are split into quantile
    bins. Point masses such as ``capital-gain == 0`` therefore keep their own
    representative value instead of being averaged into a bin that spans a
    large part of the range.

    Args:
        col: Continuous column.
        n_bins: Target number of bins.

    Returns:
        Integer bin codes, contiguous from zero.
    """
    counts = col.value_counts()
    dominant = sorted(counts[counts >= DOMINANT_VALUE_SHARE * len(col)].index)

    codes = pd.Series(0, index=col.index, dtype=int)
    for code, value in enumerate(dominant):
        codes[col == value] = code

    rest = col[~col.isin(dominant)]
    if len(rest) > 0:
        quantile_bins = max(1, n_bins - len(dominant))
        rest_codes = pd.qcut(rest, q=quantile_bins, labels=False, duplicates="drop")
        codes[rest.index] = rest_codes.astype(int) + len(dominant)

    return codes


def _discretize_for_flai(df: pd.DataFrame, n_bins: int = 5) -> tuple:
    """Discretize high-cardinality columns into quantile bins.
    Columns with ≤ MAX_CARD unique values are left intact.

    String/categorical columns are label-encoded so FLAI can process them.
    The encoding mapping is returned so callers can restore semantic labels
    in the generated data.

    Continuous columns are binned by ``_bin_continuous`` and each bin is
    represented by the empirical median of the observations falling in it.
    Skewed variables such as ``capital-gain``, where a large fraction of the
    rows share the minimum value, would otherwise be inverse-transformed to a
    value that occurs nowhere in the data. Heavily tied columns may yield fewer
    than ``n_bins`` bins; the representative values are keyed by the bin codes
    actually produced.

    Args:
        df: Input DataFrame.
        n_bins: Target number of bins for continuous high-cardinality columns.

    Returns:
        tuple[pd.DataFrame, dict, dict]: Discretized int64 DataFrame, a
            ``cat_maps`` dict mapping column name -> {int_code: original_value}
            for every column that was label-encoded from strings or kept as a
            low-cardinality integer, and a
            ``num_bin_midpoints`` dict mapping column name -> {bin_int: value}
            for every continuous column that was discretized.
    """
    df = df.copy()
    cat_maps = {}
    num_bin_midpoints = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # String column: encode to int, record inverse mapping
            categories = sorted(df[col].astype(str).unique())
            code_map = {cat: i for i, cat in enumerate(categories)}
            inv_map  = {i: cat for cat, i in code_map.items()}
            cat_maps[col] = inv_map
            df[col] = df[col].astype(str).map(code_map).astype(int)
        elif df[col].nunique() > MAX_CARD:
            # Continuous column: bin and record a representative value per bin
            binned = _bin_continuous(df[col], n_bins)
            representatives = df[col].groupby(binned).median().round().astype(int)
            num_bin_midpoints[col] = representatives.to_dict()
            df[col] = binned
        else:
            df[col] = df[col].astype(int)
            values = sorted(int(v) for v in df[col].unique())
            cat_maps[col] = {i: v for i, v in enumerate(values)}
    return df, cat_maps, num_bin_midpoints


def _restore_categories(df: pd.DataFrame, cat_maps: dict, num_bin_midpoints: dict = None) -> pd.DataFrame:
    """Restore original string labels and numeric scales in columns that were encoded.

    Args:
        df: DataFrame with integer-coded columns (as generated by FLAI).
        cat_maps: Dict {col: {int_code: original_value}} returned by
            ``_discretize_for_flai``.
        num_bin_midpoints: Dict {col: {bin_int: value}} returned by
            ``_discretize_for_flai``. If provided, continuous binned columns
            are inverse-transformed to their representative values so that
            LLMs receive interpretable numeric values instead of bin indices.
            Codes outside the recorded range are clipped to the nearest bin.

    Returns:
        DataFrame with string labels and numeric scales restored.
    """
    df = df.copy()
    for col, inv_map in cat_maps.items():
        if col in df.columns:
            # FLAI-generated codes may be floats; cast to int first
            numeric = all(isinstance(v, int) for v in inv_map.values())
            df[col] = df[col].apply(
                lambda v, m=inv_map, numeric=numeric: (
                    m.get(int(round(v)), int(round(v)) if numeric else str(int(round(v))))
                    if pd.notna(v) else v
                )
            )
    if num_bin_midpoints:
        for col, representatives in num_bin_midpoints.items():
            if col in df.columns:
                codes = sorted(representatives)
                lo, hi = codes[0], codes[-1]
                df[col] = df[col].apply(
                    lambda v, r=representatives, lo=lo, hi=hi: (
                        r[min(max(int(round(v)), lo), hi)] if pd.notna(v) else v
                    )
                )
    return df


# ---------------------------------------------------------------------------
# CAUSAL FAIR DATA GENERATION (FLAI)
# ---------------------------------------------------------------------------

def generate_fair_data_causal(df: pd.DataFrame, target_col: str, sensitive_features: list, n_samples: int = None, method: str = "bayes", save_path: Path = None, seed: int = 42) -> pd.DataFrame:
    """Generate Fair Data using the FLAI mitigated causal model.

    Args:
        df: Original DataFrame with all features.
        target_col: Name of the label column.
        sensitive_features: List of protected attributes (e.g. ['sex', 'race']).
        n_samples: Number of samples to generate. If None, uses the same size as df.
        method: Sampling method ('bayes' — only applies with FLAI backend).
        save_path: Optional path to save the resulting CSV.

    Returns:
        DataFrame with generated Fair Data.
    """
    if flai_data is None:
        raise ImportError("flai-causal or bnlearn not installed. Run: pip install flai-causal bnlearn")

    if n_samples is None:
        n_samples = len(df)

    # Discretize high-cardinality columns — FLAI builds CPD tables with a
    # cartesian product over parent values. Continuous columns with dozens of
    # unique values cause the product to reach millions of combinations.
    # cat_maps captures string→int encodings and num_bin_midpoints captures
    # bin intervals so we can restore interpretable values afterwards.
    random.seed(seed)
    np.random.seed(seed)

    df_disc, cat_maps, num_bin_midpoints = _discretize_for_flai(df, n_bins=5)
    print(f"[FLAI] Cardinalities after discretization: "
          f"{ {c: df_disc[c].nunique() for c in df_disc.columns} }")

    # Learn causal structure with controlled max_indegree
    print("[FLAI] Learning causal structure")
    graph_struct = bn.structure_learning.fit(
        df_disc, methodtype='hc', scoretype='bic', max_indegree=3, verbose=0
    )
    edges = graph_struct['model_edges']

    # Connect isolated nodes to target
    nodes_in_graph = {n for e in edges for n in e}
    isolated = [c for c in df_disc.columns if c not in nodes_in_graph and c != target_col]
    for node in isolated:
        edges = edges + [(node, target_col)]
    if isolated:
        print(f"[FLAI] Isolated nodes connected to target: {isolated}")
    print(f"[FLAI] Final edges: {edges}")

    # Causal mitigation via FLAI
    flai_dataset = flai_data.Data(df_disc, transform=False)
    graph = causal_graph.CausalGraph(
        flai_dataset, node_edge=edges, target=target_col, indepence_test=False
    )
    print("[FLAI] Mitigating causal relationships...")
    graph.mitigate_edge_relation(sensible_feature=sensitive_features)

    # Workaround FLAI bug: reattach orphan nodes after mitigate_edge_relation
    # mitigate_edge_relation() can leave nodes with no edges (orphans):
    #   - sensitive attributes lose their outgoing edges (by design)
    #   - normal nodes can become isolated if their only edge connected to a sensitive node
    # FLAI does not handle these orphans when rebuilding the DAG in mitigate_calculation_cpd(),
    # which produces: "CPD defined on variable not in the model".
    print("[FLAI] Mitigating CPDs (with FLAI bug workaround)...")
    edges_after = list(graph.graph['model_edges'])
    nodes_after = {n for e in edges_after for n in e}
    all_nodes = list(df_disc.columns)
    orphans = [n for n in all_nodes if n not in nodes_after and n != target_col]
    if orphans:
        for n in orphans:
            edges_after = edges_after + [(n, target_col)]
        graph.graph['model_edges'] = edges_after
        graph.graph['model'] = bn.make_DAG(edges_after, verbose=0)['model']
        print(f"[FLAI] Orphan nodes reconnected to target: {orphans}")
    graph.mitigate_calculation_cpd(sensible_feature=sensitive_features)

    # Renormalize all CPDs after mitigation — FLAI can leave CPDs that don't sum to 1,
    # which causes pgmpy's check_model() to raise ValueError during sampling.
    for cpd in graph.graph['model'].cpds:
        raw = np.array(cpd.values)
        n_states = raw.shape[0]
        flat = raw.reshape(n_states, -1)
        s = flat.sum(axis=0, keepdims=True)
        s[s == 0] = 1
        cpd.values = (flat / s).reshape(raw.shape)
    graph.graph['model'].check_model()

    print(f"[FLAI] Generating {n_samples} samples (method={method}, seed={seed})...")
    result = graph.generate_dataset(n_samples=n_samples, methodtype=method)
    fair_df = result.data if hasattr(result, "data") else result
    fair_df = fair_df[[c for c in df.columns if c in fair_df.columns]]

    # Restore original string labels and numeric scales so the generated
    # D2 CSV contains interpretable values (e.g. "Bachelors" not "11",
    # hours-per-week ~40 not bin index 3).
    if cat_maps or num_bin_midpoints:
        fair_df = _restore_categories(fair_df, cat_maps, num_bin_midpoints)
        if cat_maps:
            print(f"[FLAI] Restored string labels for columns: {list(cat_maps.keys())}")
        if num_bin_midpoints:
            print(f"[FLAI] Restored numeric midpoints for columns: {list(num_bin_midpoints.keys())}")

    if save_path:
        fair_df.to_csv(save_path, index=False)
        print(f"[fair_data] Fair Data saved to {save_path}")

    print(f"[fair_data] Fair Data generated: {fair_df.shape}")
    return fair_df


# ---------------------------------------------------------------------------
# SIMPLE RESAMPLING (baseline Liu et al., 2024)
# ---------------------------------------------------------------------------

def generate_resampled_data(df: pd.DataFrame, sensitive_feature: str, privileged_value: int, strategy: str = "oversample", save_path: Path = None) -> pd.DataFrame:
    """Generate balanced data via simple resampling.

    Replicates the baseline from Liu et al. (NAACL 2024): oversampling the
    minority group or undersampling the majority group.

    Args:
        df: Original DataFrame.
        sensitive_feature: Protected attribute to balance.
        privileged_value: Value of the privileged group (e.g. 1=Male).
        strategy: 'oversample' or 'undersample'.
        save_path: Optional path to save the resulting CSV.

    Returns:
        Balanced DataFrame.
    """
    priv = df[df[sensitive_feature] == privileged_value]
    unpriv = df[df[sensitive_feature] != privileged_value]

    n_priv = len(priv)
    n_unpriv = len(unpriv)

    print(f"[Resample] {sensitive_feature}: privileged={n_priv}, underprivileged={n_unpriv}")

    if strategy == "oversample":
        target_n = max(n_priv, n_unpriv)
        if n_priv > n_unpriv:
            unpriv_resampled = resample(unpriv, replace=True, n_samples=target_n, random_state=42)
            resampled_df = pd.concat([priv, unpriv_resampled])
        else:
            priv_resampled = resample(priv, replace=True, n_samples=target_n, random_state=42)
            resampled_df = pd.concat([priv_resampled, unpriv])
    elif strategy == "undersample":
        target_n = min(n_priv, n_unpriv)
        if n_priv > n_unpriv:
            priv_resampled = resample(priv, replace=False, n_samples=target_n, random_state=42)
            resampled_df = pd.concat([priv_resampled, unpriv])
        else:
            unpriv_resampled = resample(unpriv, replace=False, n_samples=target_n, random_state=42)
            resampled_df = pd.concat([priv, unpriv_resampled])
    else:
        raise ValueError(f"strategy must be 'oversample' or 'undersample', got: {strategy}")

    resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[Resample] Result: {resampled_df.shape} | "
          f"privileged={len(resampled_df[resampled_df[sensitive_feature]==privileged_value])}, "
          f"underprivileged={len(resampled_df[resampled_df[sensitive_feature]!=privileged_value])}")

    if save_path:
        resampled_df.to_csv(save_path, index=False)
        print(f"[Resample] Saved to {save_path}")

    return resampled_df


# ---------------------------------------------------------------------------
# MAKE ALL HAPPEN
# ---------------------------------------------------------------------------

def prepare_all_datasets(df: pd.DataFrame, dataset_name: str, target_col: str, sensitive_features: list, privileged_values: dict, output_dir: Path, test_size: int = 500, seed: int = 42) -> dict:
    """Generate the canonical split and the dataset versions required for the experiment.

    A single stratified train/test split of the original data is fixed first;
    every derived artifact is generated from D1_train only, so no test row
    influences discretization, structure learning, CPDs, mitigation or
    resampling. D1 and D2 test sets are not paired: D2 is an independent
    synthetic sample from the mitigated causal model, not a row-wise
    counterfactual of D1.

    Files generated in output_dir/{dataset_name}/:
    - D1_train.csv / D1_test.csv : stratified split of the original data.
    - D2_train.csv / D2_test.csv : causal Fair Data (FLAI fitted on D1_train),
      split into two disjoint sets of independently sampled records.
    - D3_train.csv               : oversampling of D1_train on the first
      protected attribute.

    Args:
        df: Preprocessed original DataFrame.
        dataset_name: Dataset name ('adult', 'compas', 'german').
        target_col: Name of the target column.
        sensitive_features: List of protected attributes.
        privileged_values: Dictionary {feature: privileged_value}.
        output_dir: Root output directory.
        test_size: Number of test instances in the canonical split.
        seed: Random seed for the canonical split.

    Returns:
        Dictionary with keys 'D1_train', 'D1_test', 'D2_train', 'D2_test',
        'D3_train' and their loaded DataFrames.
    """
    out = output_dir / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    # D1 — canonical stratified split
    d1_train_path = out / "D1_train.csv"
    d1_test_path  = out / "D1_test.csv"
    if d1_train_path.exists() and d1_test_path.exists():
        print(f"\n[D1] Split already exists, skipping: {d1_train_path}")
        df_train = pd.read_csv(d1_train_path)
        df_test  = pd.read_csv(d1_test_path)
    else:
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[target_col]
        )
        df_train = df_train.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)
        df_train.to_csv(d1_train_path, index=False)
        df_test.to_csv(d1_test_path, index=False)
        print(f"\n[D1] Canonical split saved: {len(df_train)} train / {len(df_test)} test")

    # D2 — Causal Fair Data, fitted on D1_train only
    d2_train_path = out / "D2_train.csv"
    d2_test_path  = out / "D2_test.csv"
    if d2_train_path.exists() and d2_test_path.exists():
        print(f"\n[D2] Already exists, skipping generation: {d2_train_path}")
    else:
        print("\n[D2] Generating causal Fair Data from D1_train")
        fair_df = generate_fair_data_causal(
            df=df_train,
            target_col=target_col,
            sensitive_features=sensitive_features,
            n_samples=len(df_train) + len(df_test),
            seed=seed,
        )
        fair_df.iloc[:len(df_train)].reset_index(drop=True).to_csv(d2_train_path, index=False)
        fair_df.iloc[len(df_train):].reset_index(drop=True).to_csv(d2_test_path, index=False)
        print(f"[D2] Saved: {len(df_train)} train / {len(df_test)} test synthetic samples")

    # D3 — Resampling of D1_train (uses the first protected attribute as primary)
    primary_sensitive = sensitive_features[0]
    d3_train_path = out / "D3_train.csv"
    if d3_train_path.exists():
        print(f"\n[D3] Already exists, skipping generation: {d3_train_path}")
    else:
        print("\n[D3] Generating resampled data from D1_train")
        generate_resampled_data(
            df=df_train,
            sensitive_feature=primary_sensitive,
            privileged_value=privileged_values[primary_sensitive],
            strategy="oversample",
            save_path=d3_train_path,
        )

    print(f"\n[prepare_all_datasets] Completed for '{dataset_name}'.")
    return {
        "D1_train": pd.read_csv(d1_train_path),
        "D1_test":  pd.read_csv(d1_test_path),
        "D2_train": pd.read_csv(d2_train_path),
        "D2_test":  pd.read_csv(d2_test_path),
        "D3_train": pd.read_csv(d3_train_path),
    }

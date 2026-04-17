import argparse
import json
import logging
import random
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, Draw, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
except Exception:  # pragma: no cover - depends on RDKit build
    FilterCatalog = None
    FilterCatalogParams = None


RANDOM_SEED = 42
TARGET_QUERY = "KRAS"
MAX_MOLS = 400
MAX_ANALOGS = 400
MAX_DOCK = 20
RECEPTOR = Path("protein/kras.pdbqt")
SOURCE_PDB = Path("protein/KRAS_G12C.pdb")

OUT_DIR = Path("results")
DATA_DIR = OUT_DIR / "data"
PLOTS_DIR = OUT_DIR / "plots"
MOLS_DIR = OUT_DIR / "molecules"
MODEL_DIR = OUT_DIR / "model"
DOCK_DIR = OUT_DIR / "docking_logs"
SUMMARY_FILE = OUT_DIR / "run_summary.json"

HOST = "www.ebi.ac.uk"
REST_PAGE_LIMIT = 200
REST_TIMEOUT = (10, 30)
MAX_REQUEST_RETRIES = 3
MAX_PAGES = 300
FP_BITS = 2048
VINA_EXHAUSTIVENESS = 8

DEFAULT_BOX_CENTER = (10.0, 10.0, 10.0)
DEFAULT_BOX_SIZE = (20.0, 20.0, 20.0)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VINA_TOP1_RE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.MULTILINE)
HALOGENS = {9, 17, 35, 53}
COMMON_ATOMS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
KRAS_SITE_RESIDUES = {9, 10, 12, 13, 14, 15, 16, 95, 96, 99, 103}


@dataclass
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    source: str


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype AI-guided drug discovery pipeline")
    parser.add_argument("--target", default=TARGET_QUERY, help="Target query for ChEMBL")
    parser.add_argument("--max-mols", type=int, default=MAX_MOLS, help="Maximum curated molecules")
    parser.add_argument("--max-analogs", type=int, default=MAX_ANALOGS, help="Maximum BRICS analogs")
    parser.add_argument("--max-dock", type=int, default=MAX_DOCK, help="Maximum molecules to dock")
    parser.add_argument("--cache-only", action="store_true", help="Skip ChEMBL and use local cleaned_data.csv")
    parser.add_argument("--receptor", type=Path, default=RECEPTOR, help="Receptor PDBQT path")
    parser.add_argument("--source-pdb", type=Path, default=SOURCE_PDB, help="Source PDB for docking box inference")
    parser.add_argument(
        "--vina-exhaustiveness",
        type=int,
        default=VINA_EXHAUSTIVENESS,
        help="AutoDock Vina exhaustiveness",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Manual docking box center coordinates",
    )
    parser.add_argument(
        "--size",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        help="Manual docking box sizes",
    )
    return parser.parse_args()


def make_dirs():
    for p in [DATA_DIR, PLOTS_DIR, MOLS_DIR, MODEL_DIR, DOCK_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _rows_to_df(rows, max_mols):
    if not rows:
        raise ValueError("No data fetched")

    raw = pd.DataFrame(rows)
    raw.to_csv(DATA_DIR / "raw_chembl.csv", index=False)

    df = raw.groupby("smiles", as_index=False)["pIC50"].median()
    df = df[(df["pIC50"] > 0) & (df["pIC50"] < 12)]
    df = df.sort_values("pIC50", ascending=False).head(max_mols).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable molecules after cleaning")

    df.to_csv(DATA_DIR / "cleaned_data.csv", index=False)
    logging.info("Fetched %d unique molecules", len(df))
    return df


def _load_local_cleaned_data():
    fp = DATA_DIR / "cleaned_data.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    if {"smiles", "pIC50"}.issubset(df.columns) and len(df) >= 50:
        logging.warning("Using cached local dataset: %s (%d rows)", fp, len(df))
        return df
    return None


def _host_resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def _make_retry_session():
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "DrugPipe/2.0"})
    return session


def _safe_get_json(session: requests.Session, url: str, params=None):
    last_error = None
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REST_TIMEOUT)
            content_type = (response.headers.get("Content-Type") or "").lower()
            if response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code} from {url}")
            response.raise_for_status()
            if "json" not in content_type:
                raise ValueError(f"Non-JSON response from {url}: {content_type}")
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == MAX_REQUEST_RETRIES:
                break
            wait = min(15, attempt * 2 + random.random())
            logging.warning(
                "Request failed: %s | retry %d/%d in %.1fs",
                exc,
                attempt,
                MAX_REQUEST_RETRIES,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def _resolve_target_id_rest(session: requests.Session, query: str) -> str:
    js = _safe_get_json(
        session,
        "https://www.ebi.ac.uk/chembl/api/data/target/search.json",
        params={"q": query, "limit": 50},
    )
    targets = js.get("targets", [])
    if not targets:
        raise ValueError(f"No target found for query '{query}'")

    scored = []
    for target in targets:
        pref = (target.get("pref_name") or "").upper()
        org = (target.get("organism") or "").lower()
        ttype = (target.get("target_type") or "").lower()
        score = 0
        if pref == "KRAS":
            score += 5
        if org == "homo sapiens":
            score += 3
        if ttype == "single protein":
            score += 2
        scored.append((score, target))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][1]
    logging.info(
        "Using target %s | pref_name=%s | organism=%s | type=%s",
        best["target_chembl_id"],
        best.get("pref_name"),
        best.get("organism"),
        best.get("target_type"),
    )
    return best["target_chembl_id"]


def _fetch_via_rest(target_name=TARGET_QUERY, max_mols=MAX_MOLS):
    session = _make_retry_session()
    target_id = _resolve_target_id_rest(session, target_name)

    rows = []
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": target_id,
        "standard_type": "IC50",
        "standard_units": "nM",
        "standard_relation": "=",
        "limit": REST_PAGE_LIMIT,
    }

    page_count = 0
    while url and len(rows) < max_mols * 6 and page_count < MAX_PAGES:
        page_count += 1
        js = _safe_get_json(session, url, params=params)

        for activity in js.get("activities", []):
            smiles = activity.get("canonical_smiles")
            value = activity.get("standard_value")
            if not smiles or not value:
                continue

            try:
                ic50_nm = float(value)
            except (TypeError, ValueError):
                continue
            if ic50_nm <= 0:
                continue

            canonical = canonicalize_smiles(smiles)
            if canonical is None:
                continue

            pic50 = -np.log10(ic50_nm * 1e-9)
            if np.isfinite(pic50):
                rows.append({"smiles": canonical, "pIC50": pic50})

            if len(rows) >= max_mols * 6:
                break

        next_page = js.get("page_meta", {}).get("next")
        if not next_page:
            break
        url = next_page if next_page.startswith("http") else f"https://www.ebi.ac.uk{next_page}"
        params = None

    return _rows_to_df(rows, max_mols)


def fetch_chembl_data(target_name=TARGET_QUERY, max_mols=MAX_MOLS, cache_only=False):
    cached = _load_local_cleaned_data()
    if cache_only:
        if cached is None:
            raise RuntimeError("cache-only requested, but results/data/cleaned_data.csv was not found.")
        return cached, "cache-only"

    if not _host_resolves(HOST):
        if cached is not None:
            logging.warning("Host %s does not resolve; using cached dataset.", HOST)
            return cached, "cache-no-network"
        raise RuntimeError(f"Host {HOST} does not resolve and no cached cleaned_data.csv is available.")

    try:
        return _fetch_via_rest(target_name=target_name, max_mols=max_mols), "chembl-rest"
    except Exception as exc:
        logging.warning("ChEMBL fetch failed: %s", exc)
        if cached is not None:
            return cached, "cache-fetch-fallback"
        raise RuntimeError("ChEMBL unavailable and no usable local cleaned_data.csv found.") from exc


def build_filter_catalog():
    if FilterCatalogParams is None or FilterCatalog is None:
        return None
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


PAINS_CATALOG = build_filter_catalog()


def molecular_descriptor_vector(mol):
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            QED.qed(mol),
        ],
        dtype=np.float32,
    )


def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    fp_arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    desc_arr = molecular_descriptor_vector(mol)
    return np.concatenate([fp_arr, desc_arr]).astype(np.float32)


def featurize_many(smiles_list):
    feats = []
    valid = []
    scaffolds = []
    for smiles in smiles_list:
        feature = featurize(smiles)
        if feature is None:
            continue
        feats.append(feature)
        valid.append(smiles)
        mol = Chem.MolFromSmiles(smiles)
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "")
    return np.asarray(feats, dtype=np.float32), valid, scaffolds


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def scaffold_split_indices(smiles_list, test_fraction=0.2):
    scaffold_groups = {}
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or f"NO_SCAFFOLD_{idx}"
        scaffold_groups.setdefault(scaffold, []).append(idx)

    groups = list(scaffold_groups.values())
    groups.sort(key=lambda items: (-len(items), items[0]))

    total = len(smiles_list)
    target_test = max(1, int(round(total * test_fraction)))
    test_indices = []
    for group in groups:
        if len(test_indices) >= target_test and test_indices:
            break
        test_indices.extend(group)

    test_set = sorted(set(test_indices))
    train_set = [idx for idx in range(total) if idx not in test_set]
    if not train_set or not test_set:
        raise ValueError("Unable to create scaffold split")
    return train_set, test_set


def save_parity_plot(y_true, y_pred, label, filename):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("True pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(f"Parity Plot ({label})")
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()


def train_model(df: pd.DataFrame):
    X, valid_smiles, _ = featurize_many(df["smiles"].tolist())
    y_map = dict(zip(df["smiles"], df["pIC50"]))
    y = np.asarray([y_map[smiles] for smiles in valid_smiles], dtype=np.float32)

    if len(y) < 20:
        raise ValueError("Not enough valid molecules for training")

    X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
        X,
        y,
        valid_smiles,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    random_model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    random_model.fit(X_train, y_train)
    random_preds = random_model.predict(X_test)
    random_metrics = compute_metrics(y_test, random_preds)
    save_parity_plot(y_test, random_preds, "Random Split", PLOTS_DIR / "parity_random.png")

    scaffold_train_idx, scaffold_test_idx = scaffold_split_indices(valid_smiles, test_fraction=0.2)
    scaffold_model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    scaffold_model.fit(X[scaffold_train_idx], y[scaffold_train_idx])
    scaffold_preds = scaffold_model.predict(X[scaffold_test_idx])
    scaffold_metrics = compute_metrics(y[scaffold_test_idx], scaffold_preds)
    save_parity_plot(y[scaffold_test_idx], scaffold_preds, "Scaffold Split", PLOTS_DIR / "parity.png")
    save_parity_plot(y[scaffold_test_idx], scaffold_preds, "Scaffold Split", PLOTS_DIR / "parity_scaffold.png")

    plt.figure()
    plt.hist(y, bins=20)
    plt.xlabel("pIC50")
    plt.ylabel("Count")
    plt.title("Training Label Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "distribution.png", dpi=160)
    plt.close()

    final_model = RandomForestRegressor(
        n_estimators=600,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    final_model.fit(X, y)

    metrics = {
        "n_molecules": int(len(valid_smiles)),
        "n_unique_scaffolds": int(len({MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or "" for s in valid_smiles})),
        "random_split": random_metrics,
        "scaffold_split": scaffold_metrics,
        "recommended_metric_for_reporting": "scaffold_split",
    }

    with open(MODEL_DIR / "metrics.txt", "w") as handle:
        handle.write(f"n_molecules: {metrics['n_molecules']}\n")
        handle.write(f"n_unique_scaffolds: {metrics['n_unique_scaffolds']}\n")
        handle.write(
            "random_split: "
            f"R2={random_metrics['r2']:.4f}, RMSE={random_metrics['rmse']:.4f}, "
            f"MAE={random_metrics['mae']:.4f}, n_test={random_metrics['n_test']}\n"
        )
        handle.write(
            "scaffold_split: "
            f"R2={scaffold_metrics['r2']:.4f}, RMSE={scaffold_metrics['rmse']:.4f}, "
            f"MAE={scaffold_metrics['mae']:.4f}, n_test={scaffold_metrics['n_test']}\n"
        )
        handle.write("recommended_metric_for_reporting: scaffold_split\n")

    logging.info(
        "Model performance | random R2=%.3f RMSE=%.3f | scaffold R2=%.3f RMSE=%.3f",
        random_metrics["r2"],
        random_metrics["rmse"],
        scaffold_metrics["r2"],
        scaffold_metrics["rmse"],
    )
    return final_model, metrics


def generate_brics_analogs(seed_smiles, max_analogs=MAX_ANALOGS):
    seed_set = set(seed_smiles)
    frag_mols = []
    seen_frag = set()

    for smiles in seed_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for frag_mol in BRICS.BRICSDecompose(mol, returnMols=True):
            frag_smiles = Chem.MolToSmiles(frag_mol, canonical=True)
            if frag_smiles in seen_frag:
                continue
            seen_frag.add(frag_smiles)
            frag_mols.append(frag_mol)

    random.shuffle(frag_mols)
    frag_mols = frag_mols[:300]
    if not frag_mols:
        logging.warning("No BRICS fragments generated")
        return []

    analogs = []
    seen = set(seed_set)
    for mol in BRICS.BRICSBuild(frag_mols, maxDepth=3):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in seen:
            continue
        seen.add(smiles)
        analogs.append(smiles)
        if len(analogs) >= max_analogs:
            break

    generated_df = pd.DataFrame({"smiles": analogs, "source": "BRICS"})
    generated_df.to_csv(DATA_DIR / "generated.csv", index=False)
    logging.info("Generated %d novel BRICS analogs", len(analogs))
    return analogs


def passes_quality_filters(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False

    atoms = mol.GetAtoms()
    if any(atom.GetAtomicNum() not in COMMON_ATOMS for atom in atoms):
        return False
    if any(abs(atom.GetFormalCharge()) > 2 for atom in atoms):
        return False
    if any(atom.GetAtomicNum() in HALOGENS and atom.GetTotalValence() > 1 for atom in atoms):
        return False

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    heavy = mol.GetNumHeavyAtoms()
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    qed_score = QED.qed(mol)

    if PAINS_CATALOG is not None and PAINS_CATALOG.HasMatch(mol):
        return False

    return (
        180 <= mw <= 650
        and -1 <= logp <= 6.5
        and hbd <= 6
        and hba <= 12
        and tpsa <= 180
        and 14 <= heavy <= 70
        and rot_bonds <= 12
        and 1 <= rings <= 8
        and aromatic_rings <= 5
        and frac_csp3 <= 0.8
        and qed_score >= 0.15
    )


def score_generated(model, smiles_list):
    Xg, valid, _ = featurize_many(smiles_list)
    if len(valid) == 0:
        logging.warning("Generated set has 0 featurizable molecules")
        return pd.DataFrame(columns=["smiles", "predicted_pIC50"])

    pred = model.predict(Xg)
    all_df = pd.DataFrame({"smiles": valid, "predicted_pIC50": pred}).drop_duplicates("smiles")
    all_df["passes_quality_filters"] = all_df["smiles"].map(passes_quality_filters)
    logging.info("Generated molecules: %d | featurized unique: %d", len(smiles_list), len(all_df))

    filtered = all_df[all_df["passes_quality_filters"]].copy()
    logging.info("After medicinal chemistry filters: %d", len(filtered))

    if filtered.empty:
        logging.warning("Quality filters removed everything; falling back to top predicted molecules")
        filtered = all_df.sort_values("predicted_pIC50", ascending=False).head(100).copy()

    filtered = filtered.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)
    filtered.to_csv(DATA_DIR / "generated_predictions.csv", index=False)
    return filtered


def parse_vina_top1(text: str):
    match = VINA_TOP1_RE.search(text or "")
    return float(match.group(1)) if match else None


def infer_docking_box(source_pdb: Path):
    if source_pdb.exists():
        hetero_coords = []
        site_coords = []
        with source_pdb.open() as handle:
            for line in handle:
                record = line[:6].strip()
                if record not in {"ATOM", "HETATM"}:
                    continue
                resname = line[17:20].strip()
                chain = line[21].strip()
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coord = np.array([x, y, z], dtype=float)
                if record == "HETATM" and resname not in {"HOH", "WAT", "NA", "CL", "MG", "ZN", "SO4"}:
                    hetero_coords.append(coord)
                if record == "ATOM" and chain == "A" and resseq in KRAS_SITE_RESIDUES:
                    site_coords.append(coord)

        if hetero_coords:
            coords = np.vstack(hetero_coords)
            center = coords.mean(axis=0)
            sizes = np.clip(coords.max(axis=0) - coords.min(axis=0) + 8.0, 18.0, 30.0)
            return DockingBox(*center.tolist(), *sizes.tolist(), "heteroatom-ligand")

        if site_coords:
            coords = np.vstack(site_coords)
            center = coords.mean(axis=0)
            sizes = np.clip(coords.max(axis=0) - coords.min(axis=0) + 6.0, 18.0, 26.0)
            return DockingBox(*center.tolist(), *sizes.tolist(), "kras-site-residues-approx")

    return DockingBox(*DEFAULT_BOX_CENTER, *DEFAULT_BOX_SIZE, "fallback-default")


def docking_score(smiles: str, idx: int, receptor_path: Path, docking_box: DockingBox, exhaustiveness: int):
    work = Path(tempfile.mkdtemp(prefix=f"dock_{idx}_", dir=DOCK_DIR))
    smi_file = work / f"lig_{idx}.smi"
    lig_file = work / f"lig_{idx}.pdbqt"
    out_file = work / f"out_{idx}.pdbqt"
    log_file = DOCK_DIR / f"log_{idx}.txt"

    try:
        smi_file.write_text(smiles + "\n")

        obabel = subprocess.run(
            ["obabel", str(smi_file), "-O", str(lig_file), "--gen3d"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if obabel.returncode != 0:
            log_file.write_text(f"[obabel failed]\n{obabel.stdout}\n{obabel.stderr}")
            return None

        vina = subprocess.run(
            [
                "vina",
                "--receptor",
                str(receptor_path),
                "--ligand",
                str(lig_file),
                "--center_x",
                f"{docking_box.center_x:.3f}",
                "--center_y",
                f"{docking_box.center_y:.3f}",
                "--center_z",
                f"{docking_box.center_z:.3f}",
                "--size_x",
                f"{docking_box.size_x:.3f}",
                "--size_y",
                f"{docking_box.size_y:.3f}",
                "--size_z",
                f"{docking_box.size_z:.3f}",
                "--seed",
                str(RANDOM_SEED),
                "--exhaustiveness",
                str(exhaustiveness),
                "--out",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        log_text = (vina.stdout or "") + "\n" + (vina.stderr or "")
        log_file.write_text(log_text)
        if vina.returncode != 0:
            return None
        return parse_vina_top1(log_text)
    except Exception as exc:
        log_file.write_text(f"[exception] {exc}")
        return None


def zscore(series: pd.Series):
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def rank_hits(df: pd.DataFrame):
    out = df.copy()
    out["dock_affinity"] = -out["dock_score"]
    out["z_pred"] = zscore(out["predicted_pIC50"])
    out["z_dock"] = zscore(out["dock_affinity"])
    out["final_score"] = 0.6 * out["z_pred"] + 0.4 * out["z_dock"]
    return out.sort_values("final_score", ascending=False).reset_index(drop=True)


def save_molecule_images(smiles_list, max_images=20):
    for idx, smiles in enumerate(smiles_list[:max_images]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Draw.MolToFile(mol, str(MOLS_DIR / f"mol_{idx}.png"))


def write_summary(summary_dict):
    SUMMARY_FILE.write_text(json.dumps(summary_dict, indent=2))


def main():
    args = parse_args()
    make_dirs()

    summary = {
        "target": args.target,
        "cache_only": bool(args.cache_only),
        "receptor": str(args.receptor),
        "source_pdb": str(args.source_pdb),
        "status": "started",
    }

    data, data_source = fetch_chembl_data(
        target_name=args.target,
        max_mols=args.max_mols,
        cache_only=args.cache_only,
    )
    summary["data_source"] = data_source
    summary["n_training_molecules"] = int(len(data))

    model, metrics = train_model(data)
    summary["model_metrics"] = metrics

    seeds = data.sort_values("pIC50", ascending=False)["smiles"].head(min(150, len(data))).tolist()
    generated = generate_brics_analogs(seeds, max_analogs=args.max_analogs)
    if not generated:
        logging.warning("No BRICS analogs generated; using top seed molecules as fallback candidates")
        generated = seeds[: args.max_analogs]
        pd.DataFrame({"smiles": generated, "source": "seed-fallback"}).to_csv(
            DATA_DIR / "generated.csv",
            index=False,
        )

    summary["n_generated_candidates"] = int(len(generated))

    pred_df = score_generated(model, generated)
    if pred_df.empty:
        logging.warning("No valid molecules after generation/filtering; retrying with seed molecules")
        pred_df = score_generated(model, seeds[: args.max_analogs])

    if pred_df.empty:
        logging.warning("Still no valid molecules. Writing empty ranked output.")
        pd.DataFrame(columns=["smiles", "predicted_pIC50", "dock_score", "final_score"]).to_csv(
            OUT_DIR / "top_hits.csv",
            index=False,
        )
        summary["status"] = "completed-no-candidates"
        write_summary(summary)
        print("\nPipeline finished: no valid candidates survived filtering.")
        return

    summary["n_ranked_candidates"] = int(len(pred_df))

    has_obabel = shutil.which("obabel") is not None
    has_vina = shutil.which("vina") is not None
    has_receptor = args.receptor.exists()

    if args.center and args.size:
        docking_box = DockingBox(*args.center, *args.size, "manual")
    else:
        docking_box = infer_docking_box(args.source_pdb)

    summary["docking_box"] = asdict(docking_box)
    summary["vina_exhaustiveness"] = int(args.vina_exhaustiveness)

    if not (has_obabel and has_vina and has_receptor):
        logging.warning(
            "Docking skipped (obabel=%s, vina=%s, receptor=%s)",
            has_obabel,
            has_vina,
            has_receptor,
        )
        top = pred_df.head(50).copy()
        top.to_csv(OUT_DIR / "top_hits.csv", index=False)
        save_molecule_images(top["smiles"].tolist(), max_images=20)
        summary["status"] = "completed-prediction-only"
        summary["docking_performed"] = False
        summary["top_output"] = "prediction-only ranking"
        write_summary(summary)
        print("\nPipeline finished without docking.")
        print("Reason: missing obabel, vina, or receptor PDBQT.")
        print("\nTop hits (predicted only):")
        print(top.head(10))
        return

    if docking_box.source != "manual":
        logging.warning(
            "Docking box source=%s. Validate this box against a co-crystal ligand or known pocket before reporting docking claims.",
            docking_box.source,
        )

    dock_rows = []
    for idx, row in pred_df.head(args.max_dock).iterrows():
        score = docking_score(
            row["smiles"],
            int(idx),
            args.receptor,
            docking_box,
            args.vina_exhaustiveness,
        )
        if score is None:
            continue
        dock_rows.append(
            {
                "smiles": row["smiles"],
                "predicted_pIC50": float(row["predicted_pIC50"]),
                "dock_score": float(score),
            }
        )

    if not dock_rows:
        logging.warning("No valid docking scores; writing predicted-only ranking instead")
        top = pred_df.head(50).copy()
        top.to_csv(OUT_DIR / "top_hits.csv", index=False)
        save_molecule_images(top["smiles"].tolist(), max_images=20)
        summary["status"] = "completed-docking-failed"
        summary["docking_performed"] = False
        summary["top_output"] = "prediction-only ranking after docking failure"
        write_summary(summary)
        print("\nPipeline finished without valid docking scores.")
        print("\nTop hits (predicted only):")
        print(top.head(10))
        return

    final_df = rank_hits(pd.DataFrame(dock_rows))
    final_df.to_csv(OUT_DIR / "top_hits.csv", index=False)
    save_molecule_images(final_df["smiles"].tolist(), max_images=20)

    summary["status"] = "completed-with-docking"
    summary["docking_performed"] = True
    summary["n_docked"] = int(len(dock_rows))
    summary["top_output"] = "combined prediction + docking ranking"
    write_summary(summary)

    print("\nPipeline finished with docking.")
    print("\nTop hits:")
    print(final_df.head(10))


if __name__ == "__main__":
    main()

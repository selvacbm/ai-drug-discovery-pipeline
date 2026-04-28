# 🧬 AI Drug Discovery Pipeline

An end-to-end **AI-driven drug discovery workflow** integrating molecular modeling, machine learning, and virtual screening to accelerate lead identification and optimization.

---

## 🚀 Overview

This project combines **physics-based simulations** and **data-driven models** to improve compound prioritization in drug discovery. The pipeline supports:

* Molecular featurization (SMILES → descriptors)
* Machine learning-based activity prediction (QSAR)
* Structure-based virtual screening (docking)
* Ranking using hybrid ML + physics-based scoring

---

## 🧠 Key Features

* 🔬 Integration of **molecular dynamics (MD)** and **machine learning**
* ⚡ High-throughput screening of **10⁴–10⁵ compounds**
* 📊 Automated data processing and model training
* 🧪 Designed for **KRAS and enzyme targets**
* 🔁 Modular and scalable pipeline

---

## 🛠️ Tech Stack

* **Programming:** Python
* **ML Libraries:** scikit-learn, NumPy, pandas
* **Cheminformatics:** RDKit
* **Simulation Tools:** GROMACS, AMBER
* **Docking:** AutoDock Vina
* **Environment:** Linux / HPC systems

---

## 📂 Project Structure

```
ai-drug-discovery-pipeline/
│── data/           # Input datasets (SMILES, descriptors)
│── scripts/        # Pipeline scripts (ML, docking, processing)
│── models/         # Trained ML models
│── results/        # Output predictions and rankings
│── README.md
│── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone repository

```bash
git clone https://github.com/selvacbm/ai-drug-discovery-pipeline.git
cd ai-drug-discovery-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run pipeline

```bash
python run_pipeline.py
```

---

## 📊 Workflow

1. Input molecular data (SMILES)
2. Generate descriptors (RDKit)
3. Train ML model (QSAR)
4. Perform docking (AutoDock Vina)
5. Rank compounds using hybrid scoring

---

## 🎯 Applications

* Drug discovery (KRAS inhibitors)
* Enzyme engineering
* Nano-QSAR/QSPR modeling
* Molecular design and optimization

---

## 📌 Future Improvements

* Deep learning models (Graph Neural Networks)
* Enhanced sampling integration (free energy methods)
* Active learning for iterative optimization

---

## 👨‍🔬 Author

**Selvaraj Sengottiyan**
Computational Chemist | AI-driven Drug Discovery | Molecular Modeling

---

## 📄 License

MIT License

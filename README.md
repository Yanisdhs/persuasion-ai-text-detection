# DSAI Project: Persuasion, AI Detection & Argument Generation

This repository contains the source code for our project developed as part of the **DSAI (Data Science & Artificial Intelligence) program** at **Télécom Paris**.

The project focuses on three complementary tasks:

- **Argument analysis and persuasion prediction** using the **Winning Argument Corpus (WAC)** dataset.
- **AI-generated text detection** using the **GriD, HC3 and M4GT** datasets.
- **Persuasive argument generation** using **Large Language Models (LLMs)**.

The project was supervised by **Mathieu LABEAU** and **Pierre FIHEY**.

---

## 1. Project Structure

```text
Projet-DSAI/
│
├── configs/                    # Hydra configuration files (YAML)
│   ├── dataset/                # wac, hc3, m4gt, grid
│   ├── encoder/                # tfidf, w2v, roberta, features
│   ├── model/                  # svm
│   └── generation/             # Axis 3 generation configuration
│
├── src/
│   ├── data/                   # Data loading and preprocessing (WAC, HC3, M4GT)
│   ├── features/               # Feature extraction (stylistic, Jaccard)
│   ├── encoders/               # Encoders: TF-IDF, Word2Vec, RoBERTa
│   ├── models/                 # Classification models (SVM)
│   └── generation/             # Generation logic (Best-of-N, prompt engineering)
│
├── report/                     # report of the project
├── datasets/                   # Raw datasets
├── outputs/                    # Hydra outputs and generated logs
│
├── train.py                    # SVM training (Axes 1 & 2)
├── finetune_roberta.py         # RoBERTa fine-tuning (Axis 2 - M4GT)
├── finetune.py                 # LLM fine-tuning (Axis 3 - LoRA/bitsandbytes)
├── generate.py                 # Argument generation and final report creation
├── evaluate.py                 # Evaluation and visualization (PCA / t-SNE)
│
├── scripts/                    # Utility scripts (dataset download, PDF generation)
├── projetSD.tex                # LaTeX source of the project report
└── requirements.txt            # Python dependencies
```

---

## 2. Environment Setup

```bash
# Create a virtual environment
python -m venv env

# Activate on Windows
.\env\Scripts\activate

# Activate on Linux / macOS
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Download and format the required datasets:

```bash
python -X utf8 scripts/download_datasets.py --all
```

---

## 3. Running the Experiments

The project uses **Hydra** for experiment configuration, allowing parameters to be modified directly from the command line.

### Axis 1: Persuasion Prediction — Winning Argument Corpus

Training an SVM to predict whether an argument successfully changes a user's opinion:

```bash
python train.py dataset=wac encoder=features
```

### Axis 2: AI-Generated Text Detection — Human vs. AI

Training models to distinguish human-written text from AI-generated text.

```bash
# Baseline using SVM and TF-IDF on the HC3 dataset
python train.py dataset=hc3 encoder=tfidf

# Full RoBERTa fine-tuning on the M4GT dataset
python finetune_roberta.py dataset=m4gt
```

### Axis 3: Strategic Argument Generation

The generation pipeline uses the models developed in Axes 1 and 2 to guide the generation of persuasive text.

```bash
# 1. Fine-tune the LLM on winning arguments
python finetune.py llm=gpt2

# 2. Best-of-N generation and evaluation
python generate.py strategy=best_of_n \
    llm.model_id=outputs/finetuned_gpt2/ \
    axe1.model_path=axe1_svm_features_wac.pkl \
    axe1.encoder_name=features \
    axe2.model_path=outputs/roberta_finetuned_m4gt/best_model \
    axe2.encoder_name=roberta
```

After generation, an HTML report (`generation_report.html`) and a CSV file are produced containing the generated texts and their respective scores.

---

## 4. Team

### Project Members

- **Yanis DAHASSE**
- **Tristan JIN**
- **Wassim SMATI**

### Contributions

#### Yanis DAHASSE — NLP & Transformer Modeling

I primarily contributed to **Axis 1: Persuasion Prediction**, with a focus on **data preparation, bias handling and Transformer-based modeling**.

- **Dataset Engineering:** cleaned and formatted the CMV dataset and prepared the data for modeling.
- **Data Splitting:** designed the train/validation/test split **by post identifier**, ensuring that posts from the same discussion did not leak across different splits.
- **Bias Handling:** investigated and addressed **position bias** in the dataset.
- **Transformer Modeling:** designed and implemented a **RoBERTa Cross-Encoder architecture** for pairwise argument classification.
- **Fine-Tuning & Experimentation:** fine-tuned RoBERTa and explored several **DeBERTa-based approaches**, analyzing their performance and limitations.
- **Results:** the RoBERTa Cross-Encoder achieved the **best result obtained on Axis 1, with 72% accuracy**.
- **Research:** conducted a literature review of reference works related to computational argumentation and persuasion.
- **Project Organization:** developed and maintained the project timeline and experimental roadmap.

#### Tristan JIN — NLP & Feature-Based Modeling

Tristan primarily contributed to **Axis 1**, with a focus on dataset exploration, classical NLP approaches and feature-based modeling.

- Conducted the literature review and investigated the **WAC dataset**.
- Explored the dataset and investigated different data cleaning approaches.
- Implemented and evaluated **Word2Vec, TF-IDF and RoBERTa-based encoders** combined with SVM classification.
- Initially investigated **Logistic Regression** as an alternative classifier.
- Implemented the features described in the reference paper.
- Conducted **SHAP-based model analysis**.
- Implemented and evaluated the **pairwise classification approach**.
- Explored RoBERTa fine-tuning approaches.

#### Wassim SMATI — AI Detection & LLM Generation

Wassim was primarily responsible for **Axis 2: AI-Generated Text Detection** and **Axis 3: Strategic Argument Generation**.

- Designed the project's **Hydra-based architecture and configuration management**, supporting reproducible and scalable experiments.
- Developed and evaluated AI-generated text detection approaches on the **HC3, GriD and M4GT** datasets.
- Implemented and compared multiple text representations, including **TF-IDF, Word2Vec and RoBERTa-based Sentence-Transformer embeddings**.
- Developed a **RoBERTa fine-tuning pipeline** using the Hugging Face Trainer API.
- Fine-tuned **GPT-2 and Qwen 2.5 3B** using **QLoRA** under hardware constraints.
- Implemented **Best-of-N generation** and prompt engineering strategies.
- Developed interpretability tools using **t-SNE and UMAP** to compare generated and human-written text.

---

## 5. Technologies

- **Python**
- **PyTorch**
- **Scikit-learn**
- **Hugging Face Transformers**
- **RoBERTa**
- **DeBERTa**
- **Sentence Transformers**
- **TF-IDF**
- **Word2Vec**
- **SVM**
- **LLMs**
- **QLoRA**
- **LoRA**
- **Hydra**
- **SHAP**
- **t-SNE / UMAP**
- **Git / GitHub**

---

## 6. Organization & Computing Resources

The project was developed collaboratively using **GitHub** for version control and project organization.

Computationally intensive experiments were run on the **GPU servers provided by Télécom Paris**.

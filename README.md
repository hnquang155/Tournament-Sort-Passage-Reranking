# README

## Step to Reproduce the Code

**Note:** This code is designed to run on the **Deakin Compute Cluster**.

Due to limited Google Drive storage, all required files are available in a **OneDrive link**.

## Quick Start

### Step 1: Install and Verify Python Version
Ensure you have **Python 3.11 – 3.12** installed.

### Step 2: Install Requirements
Run the following command:
```bash
pip install -r requirements.txt
```
If you encounter any installation errors, follow the command prompt guidance or install the dependencies one by one:
```bash
pip install torch==2.5.1 \
            transformers==4.46.3 \
            pyserini==0.43.0 \
            ir-datasets==0.5.5 \
            openai==1.55.3 \
            tiktoken==0.8.0 \
            accelerate==1.1.1
```

### Step 3: Run the Experiment
Open the terminal and execute:
```bash
python new_experiment.py \
  --model_name "google/flan-t5-large" \
  --model_short_name "flant5" \
  --method_wise "setwise" \
  --scoring "likelihood" \
  --sort_method "tournament" \
  --r_tournament 1 \
  --shuffle_ranking "original" \
  --parent_dataset "beir" \
  --dataset "trec-covid" \
  --retrieve_step "bm25-flat" \
  --hits 100 \
  --query_length 32 \
  --passage_length 100 \
  --num_child 5
```

Alternatively, you can run the **experiment_job.sh** script on the **Deakin Compute Cluster**.

### Pre-run Checklist
- Ensure the **Hugging Face models** (FlanT5, Vicuna) are available in `llmranker`.
- Verify the dataset is stored in the `datasets` directory (including `qrels`, `corpus`, and `queries` files).
- Confirm retrieval step results are available in `Retrieve Results`.

## File Directory Explanation

### 1. `datasets/` - IR Benchmark Datasets
- Contains parent datasets: **BEIR** and **MSMARCO**.
- **BEIR**: Includes multiple sub-datasets used in the project.
- **MSMARCO**: Contains **TREC DL 2019** and **TREC DL 2020**.

### 2. `llmrankers/` - Ranking Models
- Contains **listwise** and **setwise** rankers.
- The **tournament sorting** method is implemented in `setwise` rankers.
- Currently supports **FlanT5** and **LLaMA2** series.

### 3. `Rerank Evaluation/` - Evaluation Metrics
- Stores evaluation results generated after running `new_experiment.py`.

### 4. `Retrieve Results/` - First-stage Retrieval Results
- Contains retrieval results in **Pyserini** format.
- Naming convention: `run {parent_dataset}.{retrieve_step}.{dataset}.txt`
- **Pyserini IR Toolkit** is used for the first ranking stage.

Refer to the following links for further details:
- **[Pyserini Installation & Usage](https://github.com/castorini/pyserini)**
- **[BEIR Retrieval Reproduction](https://github.com/beir-cellar/beir)**
- **[MSMARCO v1 Reproduction](https://github.com/microsoft/MSMARCO-Passage-Ranking)**

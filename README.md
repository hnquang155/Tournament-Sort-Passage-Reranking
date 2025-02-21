Step to reproduce the code
*Please note that this code works under Deakin Compute Cluster.
*Due to limit storage of Google Drive, I put all the files in one drive link
Quick Start
Step 1: Install and verify Python version: 3.11 – 3.12
Step 2: Install requirement
pip install -r requirements.txt
There could be some errors during installing the requirements, but it would be quick fix if you follow the guide on the command prompt. Otherwise, you can install one by one:
torch==2.5.1
transformers== 4.46.3
pyserini==0.43.0
ir-datasets==0.5.5
openai==1.55.3
tiktoken==0.8.0
accelerate==1.1.1
Step 3: Open the terminal and run the file new_experiment.py
Example:
python  new_experiment.py \
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

Else, you can run the file experiment_job.sh on Deakin Compute Cluster
Note: Before you run, it is recommended to check:
-	The hugging face model (FlanT5, Vicuna) is available in llmranker
-	You have the dataset saved in datasets model. This includes qrels, corpus and queries files
-	You have the result of retrieval step in Retrieve Results

File Directory Explanation
1. datasets folder
The folders include IR benchmark datasets. At this stage, we have parents’ dataset: BEIR and MSMARCO
1.1. BEIR
In BEIR, we have some sub datasets to be used in the projects
Example: 
 
1.2. MSMARCO
We have TREC DL 2019 and TREC DL 2020 in the folder at this stage.
2. llmrankers
The folder contains listwise and setwise rankers. The proposed method: tournament soring is built inside setwise rankers object
At this stage, the code is only built for FlanT5 and Llma2 series
3. Rerank Evaluation
The folder contains the metric and evaluation of each passage reranking method. The file is created after running new_experiment.py
4. Retrieve Results
This folder contains the first step retrieve result. Each file has the formant of Pyserini result
run {parent dataset}.{retrieve_step}.{dataset}.txt
For convenience, the paper relies on pyserini IR toolkit to get first ranking
Refer to this link on how to install and work on Pyserini
Refer to this for BEIR retrieval result reproduction and MSMARCO v1 for TREC DL 2019 and TREC DL 2020 reproduction


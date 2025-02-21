from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.listwise import ListwiseLlmRanker
from llmrankers.rankers import SearchResult
import time
import jsonlines
import random

import json
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

###################################################
############### Arguments to change ###############
###################################################

# The model name on hugging face. We only accept flant5 and llama series at the moment
# google/flan-t5-large
# lmsys/vicuna-7b-v1.5
model_name = "google/flan-t5-large"

#### Model name to save into the file
# flant5 or vicuna
model_short_name = "flant5"

### The method listwise or setwise
method_wise = "setwise"

### Sorting method in setwise. Choose in the set { "heapsort", "bubblesort", "tournamet"}
sort_method = "tournament"

### The parameter r in tournament sorting
r_tournament=1

# The shuffle method to test on initial positional bias. Choose in the set {None, "random", "inverse"}
shuffle_ranking = None

# Benchmark Dataset and Retrieval Selection
# Refer to Retrieve Results folder, the file has name format run.{PARENT DATASET}.{retrieve_step}.{DATASET}.txt

# Choose parent dataset. Choose in the set {"beir", "msmarco-v1-passage"}
PARENT_DATASET = "beir"

# Choose main dataset. For now, choose in the set {"trec-covid", "dl19", "dl20", "scifact"}
DATASET = "trec-covid" 

# Choose retrieval step. Refer to Retrieve Results to see the available retrieval for each dataset
retrieve_step = "bm25-flat"

# Number of passages to proceed into reranking process
hits = 100

# The number of character in query to be chunked due to tonkenization limit in LLM
query_length = 32

# The number of character in passage to be chunked due to tokeniation limit in LLM
passage_length = 100

# The sliding window size (listwise) or number of passages (childs) proceeded in the sorting algorithsm in one loop
num_child = 5

###################################################
############### Arguments to change ###############
###################################################

print("========================")
print("The method is:" , method_wise, sort_method, r_tournament)

if DATASET == "dl19":
    sub_trec_dl_dataset = "19"
elif DATASET == "dl20":
    sub_trec_dl_dataset = "19"
else:
    sub_trec_dl_dataset = None


if method_wise == "listwise":
########## Listwise ###########################
    ranker = ListwiseLlmRanker(model_name_or_path=model_name,
                            tokenizer_name_or_path=model_name,
                            device='cuda',
                            scoring='likelihood',
                            window_size=num_child,
                            step_size=2,
                            num_repeat=5
                            )
else:
########## Setwise ##############################
    ranker = SetwiseLlmRanker(model_name_or_path=model_name,
                            tokenizer_name_or_path=model_name,
                            device='cuda',
                            num_child=num_child,
                            scoring='likelihood',
                            method = sort_method,
                            k=10,
                            r_tournament=r_tournament)

if sort_method == "tournament":
    sort_method = sort_method + "-" + str(r_tournament)
if method_wise == "setwise":
    method_wise = method_wise + "-" + sort_method
if shuffle_ranking != None:
    method_wise = method_wise + shuffle_ranking


# ## Arguments

if PARENT_DATASET != "msmarco-v1-passage":

#################################### BEIR #############################################
    # Load corpus, queries and qrels
    data_path = f"datasets/{DATASET}"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Load BM25 Retrieve file
    result_file = f'Retrieve Results/run.beir.{retrieve_step}.{DATASET}.txt'  # path to your results file


#################################### TREC DL  #############################################
else:
    qrel_file = f'datasets/trec dl 20{sub_trec_dl_dataset}/qrels/test.tsv'  # path to your Qrel file
    queries_file = f'datasets/trec dl 20{sub_trec_dl_dataset}/msmarco-test20{sub_trec_dl_dataset}-queries.tsv' # path to your queries file
    result_file = f'Retrieve Results/run.msmarco-v1-passage.{retrieve_step}.{DATASET}.txt'  # path to your results file

    with open(qrel_file) as f:
        qrels = {}
        for line in f:
            query_id,  _, doc_id,  score = line.split(" ")
            try:
                #query_id = int(query_id)
                score = int(score)
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
            except:
                continue

    with open(queries_file) as f:
        queries = {}
        for line in f:
            query_id,  doc = line.split("\t")
            try:
                queries[query_id] = doc
            except:
                continue


    with jsonlines.open('datasets/msmarco/corpus.jsonl') as reader:
        corpus = {}
        for obj in reader:
            if obj["_id"] not in corpus:
                    corpus[obj["_id"]] = {}
            corpus[obj["_id"]]["text"] = obj["text"]  # obj is a dictionary containing passage data
            corpus[obj["_id"]]["title"] = obj["title"] 

##################################################################################



# Truncate query
query_map = {}
for qid in queries:
    text = queries[qid]
    query_map[qid] = ranker.truncate(text,query_length)
#docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')

#Truncate text
first_stage_rankings = []
with open(result_file, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= hits:
                continue
          
        
            #data = json.loads(corpus.doc(docid).raw())

            data = corpus[docid]
            text = data['text']
            if 'title' in data:
                text = f'{data["title"]} {text}'
            text = ranker.truncate(text, passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:hits]))

start = time.time()

# Running listwise rerank
reranked_results = []
for qid, query, ranking in tqdm(first_stage_rankings):
    if shuffle_ranking == 'random':
        random.shuffle(ranking)
    elif shuffle_ranking == 'inverse':
        ranking = ranking[::-1]
    #qid = "1"  # input query id
    query = queries[qid]

    #docs = [SearchResult(docid=docid, text=corpus[docid]["text"], score=first_stage_rankings[qid][docid]) for docid in first_stage_rankings[qid]]
    
    reranked_results.append((qid, query, ranker.rerank(query, ranking)))

end = time.time()

# Write the rerank results
def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1

save_path = f"Rerank Results/run.{method_wise}.likelihood.{model_short_name}.{retrieve_step}.{DATASET}.txt"
write_run_file(save_path, reranked_results, 'LLMRankers')     


# Load Reranking File Retrieve file
with open(save_path) as f:
    results = {}
    for line in f:
        query_id, _, doc_id, rank, score, _ = line.split()
        #query_id = int(query_id)
        score = float(score)
        if query_id not in results:
            results[query_id] = {}
        results[query_id][doc_id] = score


#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
retriever = EvaluateRetrieval(score_function="dot")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr@k")

data = {
    "ndcg": ndcg,
    "recall": recall,
    "precision": precision,
    "mrr": mrr,
    "time": end-start
}

# Write to a .txt file
file_path = f"Rerank Evaluation/run.{method_wise}.likelihood.{model_short_name}.{retrieve_step}.{DATASET}.metrics.txt"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)


# Open a new file
# file1 = open("env_info.txt", "a")
# file1.write("Get the first doc: ")
# file1.write(str(ranker.rerank(query, docs)[0]))

# # Close the file
# file1.close()

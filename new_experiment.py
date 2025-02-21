import argparse
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
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Set arguments for the ranking process.")
    # The model name on hugging face. We only accept flant5 and llama series at the moment
    # google/flan-t5-large
    # lmsys/vicuna-7b-v1.5
    parser.add_argument("--model_name", type=str, required=True, help="Model name on Hugging Face.")

    #### Model name to save into the file
    # flant5 or vicuna
    parser.add_argument("--model_short_name", type=str, choices=["flant5", "vicuna"], required=True, help="Short model name to save into the file.")
    
    parser.add_argument("--method_wise", type=str, choices=["listwise", "setwise"], required=True, help="Method: listwise or setwise.")
    parser.add_argument("--scoring", type=str, choices=["generation", "likelihood"], required=True, help="Method: Generation or logit mode")
    parser.add_argument("--sort_method", type=str, choices=["heapsort", "bubblesort", "tournament"], default="tournament", help="Sorting method for setwise.")
    parser.add_argument("--r_tournament", type=int, default=1, help="Parameter r in tournament sorting.")
    parser.add_argument("--shuffle_ranking", type=str, choices=["original", "random", "inverse"], default=None, help="Shuffle method for positional bias testing.")
    
    # Benchmark Dataset and Retrieval Selection
    # Refer to Retrieve Results folder, the file has name format run.{parent_dataset}.{retrieve_step}.{dataset}.txt

    parser.add_argument("--parent_dataset", type=str, choices=["beir", "msmarco-v1-passage"], required=True, help="Parent dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Main dataset.")
    parser.add_argument("--retrieve_step", type=str, required=True, help="Retrieval step.")
    parser.add_argument("--hits", type=int, default=100, help="Number of passages to rerank.")
    parser.add_argument("--query_length", type=int, default=32, help="Query character limit.")
    parser.add_argument("--passage_length", type=int, default=100, help="Passage character limit.")
    parser.add_argument("--num_child", type=int, default=5, help="Sliding window size or number of passages.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("========================")
    print("The method is:", args.method_wise, args.sort_method, args.r_tournament)

    if args.dataset == "dl19":
        sub_trec_dl_dataset = "19"
    elif args.dataset == "dl20":
        sub_trec_dl_dataset = "20"
    else:
        sub_trec_dl_dataset = None

    if args.method_wise == "listwise":
        ranker = ListwiseLlmRanker(
            model_name_or_path=args.model_name,
            tokenizer_name_or_path=args.model_name,
            device='cuda',
            scoring=args.scoring,
            window_size=args.num_child,
            step_size=2,
            num_repeat=5
        )
    else:
        ranker = SetwiseLlmRanker(
            model_name_or_path=args.model_name,
            tokenizer_name_or_path=args.model_name,
            device='cuda',
            num_child=args.num_child,
            scoring=args.scoring,
            method=args.sort_method,
            k=10,
            r_tournament=args.r_tournament
        )

    if args.sort_method == "tournament":
        args.sort_method = args.sort_method + "-" + str(args.r_tournament)
    if args.method_wise == "setwise":
        args.method_wise = args.method_wise + "-" + args.sort_method
    if args.shuffle_ranking != "original":
        args.method_wise = args.method_wise + args.shuffle_ranking

    if args.parent_dataset != "msmarco-v1-passage":
        data_path = f"datasets/{args.dataset}"
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        result_file = f'Retrieve Results/run.beir.{args.retrieve_step}.{args.dataset}.txt'
    else:
        qrel_file = f'datasets/trec dl 20{sub_trec_dl_dataset}/qrels/test.tsv'
        queries_file = f'datasets/trec dl 20{sub_trec_dl_dataset}/msmarco-test20{sub_trec_dl_dataset}-queries.tsv'
        result_file = f'Retrieve Results/run.msmarco-v1-passage.{args.retrieve_step}.{args.dataset}.txt'

        with open(qrel_file) as f:
            qrels = {}
            for line in f:
                query_id, _, doc_id, score = line.split(" ")
                try:
                    score = int(score)
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = score
                except:
                    continue

        with open(queries_file) as f:
            queries = {}
            for line in f:
                query_id, doc = line.split("\t")
                try:
                    queries[query_id] = doc
                except:
                    continue

        with jsonlines.open('datasets/msmarco/corpus.jsonl') as reader:
            corpus = {}
            for obj in reader:
                if obj["_id"] not in corpus:
                    corpus[obj["_id"]] = {}
                corpus[obj["_id"]]["text"] = obj["text"]
                corpus[obj["_id"]]["title"] = obj["title"]

    query_map = {}
    for qid in queries:
        text = queries[qid]
        query_map[qid] = ranker.truncate(text, args.query_length)

    first_stage_rankings = []
    with open(result_file, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.hits:
                continue

            data = corpus[docid]
            text = data['text']
            if 'title' in data:
                text = f'{data["title"]} {text}'
            text = ranker.truncate(text, args.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.hits]))

    start = time.time()
    reranked_results = []
    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.shuffle_ranking == 'random':
            random.shuffle(ranking)
        elif args.shuffle_ranking == 'inverse':
            ranking = ranking[::-1]

        reranked_results.append((qid, query, ranker.rerank(query, ranking)))

    end = time.time()
    save_path_result = f"Rerank Results/{args.dataset}/run.{args.method_wise}.{args.scoring}.{args.model_short_name}.{args.retrieve_step}.{args.dataset}.txt"

    # Ensure the folder exists
    os.makedirs(os.path.dirname(save_path_result), exist_ok=True)

    with open(save_path_result, 'w') as f:
        for qid, _, ranking in reranked_results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{'LLMRankers'}\n")
                rank += 1



    with open(save_path_result) as f:
        results = {}
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.split()
            score = float(score)
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = score

    retriever = EvaluateRetrieval(score_function="dot")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr@k")

    data = {
        "ndcg": ndcg,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "time": end - start
    }
    
    save_path_evaluation= f"Rerank Evaluation/{args.dataset}/run.{args.method_wise}.{args.scoring}.{args.model_short_name}.{args.retrieve_step}.{args.dataset}.metrics.txt"
    os.makedirs(os.path.dirname(save_path_evaluation), exist_ok=True)

    with open(save_path_evaluation, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

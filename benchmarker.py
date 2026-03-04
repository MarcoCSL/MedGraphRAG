from pathlib import Path
import numpy as np
import statistics
import argparse
import json
from tester_utils.benchmarker_utils import QADataset, locate_answer, locate_answer4pub_llama


def evaluate(num_tests, dataset, save_dir, split="test", locate_fun=locate_answer):
	flag = False
	pred = []
	answer_list = ["A", "B", "C", "D"]
	# Crea dizionario {'A': 0, 'B': 1, 'C': 2, 'D': 3}
	answer2idx = {ans:i for i, ans in enumerate(answer_list)}
	
	intervallo = range(num_tests) if num_tests > 0 else range(len(dataset))
	
	for q_idx in intervallo:
		fpath = save_dir / (split + "_" + dataset.index[q_idx] + ".json")
		answers = []
		
		for it in json.load(open(fpath))[:1]:
			answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
		
		# answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
		answers = [ans for ans in answers if ans != "NA"]
		if len(answers) == 0:
			pred.append(-1)
			continue
		
		ans = statistics.mode(answers)
		
		if ans in answer_list:
			pred.append(answer_list.index(ans))
		else:
			pred.append(-1)
	
	truth = [answer2idx[item['answer']] for item in dataset]
	
	if len(pred) < len(truth):
		truth = truth[:len(pred)]
		flag = True
	
	acc = (np.array(truth) == np.array(pred)).mean()
	std = np.sqrt(acc * (1-acc) / len(truth))
	
	return acc, std, flag

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--llm_name", type=str, default="openai/gpt-oss-20b")
	parser.add_argument("--mode", type=str, choices=["cot", "rag", "grag"], default="cot")
	parser.add_argument("--k", type=int, default=10)
	parser.add_argument("--corpus_name", type=str, default="Textbooks")
	parser.add_argument("--retriever_name", type=str, default="MedCPT")
	parser.add_argument("--nexp", type=int, default=0)

	args = parser.parse_args()

	LLM_NAME = args.llm_name
	MODE = args.mode
	K = args.k
	CORPUS = args.corpus_name
	num_tests = args.nexp
	RETRIEVER_NAME = args.retriever_name

	BASE_DIR = Path(__file__).resolve().parent

	if num_tests > 0:
		RESULT_DIR = BASE_DIR / "predictions" / ("my_predictions_" + str(num_tests))
	else:
		RESULT_DIR = BASE_DIR / "predictions" / "my_predictions_all"

	BENCHMARK = BASE_DIR / "benchmark_jsons" / ("benchmark_" +  str(num_tests) + ".json")
	
	DATASET_NAMES = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
	DATASETS = {key:QADataset(key, BENCHMARK) for key in DATASET_NAMES}
	scores = []

	for dataset_name in DATASET_NAMES:
		print("[{:s}] ".format(dataset_name), end="")
		
		split = "test"
		if dataset_name == "medmcqa":
			split = "dev"

		if MODE == "grag":
			save_dir = RESULT_DIR / dataset_name / "grag" / LLM_NAME
		elif MODE == "rag":
			save_dir = RESULT_DIR / dataset_name / ("rag_" + str(K)) / LLM_NAME / CORPUS / RETRIEVER_NAME
		else:
			save_dir = RESULT_DIR / dataset_name / "cot" / LLM_NAME
		
		if save_dir.exists():
			if "pmc_llama" in LLM_NAME.lower():
				acc, std, flag = evaluate(num_tests, DATASETS[dataset_name], save_dir, split, locate_answer4pub_llama)
			else:
				acc, std, flag = evaluate(num_tests, DATASETS[dataset_name], save_dir, split)
			
			scores.append(acc)
			print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
			
			if flag:
				print(" (NOT COMPLETED)")
			else:
				print("")
		else:
			print("NOT STARTED.")

	if len(scores) > 0:
		print("[Average] mean acc: {:.4f}".format(sum(scores) / len(scores)))

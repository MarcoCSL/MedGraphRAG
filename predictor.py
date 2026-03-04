from main_src.medgraphrag import MedGraphRAG, Mode
from tester_utils.benchmarker_utils import QADataset
from tester_utils.predictor_utils import benchmark_cloner, missing_benchmark_cloner, test_and_write

import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path
import argparse


def run_test(mode, result_dir, datasets, available_dataset_names, llm_name, corpus, retriever, k, p4p):
	modello = MedGraphRAG(llm_name, mode, retriever, corpus)
	test_and_write(
		modello,
		mode,
		result_dir,
		datasets,
		available_dataset_names,
		llm_name,
		corpus,
		retriever,
		k,
		p4p
	)

"""
Modelli:
"axiong/PMC_LLaMA_13B"
"openai/gpt-oss-20b"
"epfl-llm/meditron-70b"
"mistralai/Mixtral-8x7B-Instruct-v0.1"
"meta-llama/Meta-Llama-3-70B-Instruct"
"meta-llama/Llama-2-70b-chat-hf"
"meta-llama/Llama-3.1-8B-Instruct"
"""

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--llm_name", type=str, default="openai/gpt-oss-20b")
	parser.add_argument("--nexp", type=int, default=100)
	parser.add_argument("--k", type=int, default=10)
	parser.add_argument("--p4p", type=int, default=2)
	parser.add_argument("--corpus_name", type=str, default="Textbooks")
	parser.add_argument("--retriever_name", type=str, default="MedCPT")
	parser.add_argument("--mode", type=str, choices=["cot", "rag", "grag"], default=argparse.SUPPRESS)
	parser.add_argument("--redo_all", dest="redo_all", action="store_true")
	parser.set_defaults(redo_all=False)
	args = parser.parse_args()

	llm_name = args.llm_name
	num_tests = args.nexp  # Domande per dataset
	k = args.k if args.k >= 1 else 10
	p4p = args.p4p if args.p4p >= 1 else 2
	corpus = args.corpus_name
	retriever = args.retriever_name
	mode = args.mode
	redo_all = args.redo_all

	if hasattr(args, "mode"):
		mode = args.mode
		parallel = False
	else:
		parallel = True

	base_dir = Path(__file__).resolve().parent
	benchmark = base_dir / "benchmark_jsons" / "benchmark.json"
	dataset_names = ["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"]

	result_dir = base_dir / "predictions" / ("my_predictions_" + (str(num_tests) if num_tests > 0 else "all"))
	result_dir.mkdir(parents=True, exist_ok=True)

	if redo_all:
		cot_benchmark = rag_benchmark = grag_benchmark = benchmark = benchmark_cloner(benchmark, num_tests)
		av_datasets = cot_av_datasets = rag_av_datasets = grag_av_datasets = dataset_names
	else:
		if not parallel:
			benchmark, av_datasets = missing_benchmark_cloner(benchmark, num_tests, result_dir, mode, llm_name, corpus, retriever, k, dataset_names)
		else:
			cot_benchmark, cot_av_datasets = missing_benchmark_cloner(benchmark, num_tests, result_dir, Mode.COT, llm_name, corpus, retriever, k, dataset_names)
			rag_benchmark, rag_av_datasets = missing_benchmark_cloner(benchmark, num_tests, result_dir, Mode.RAG, llm_name, corpus, retriever, k, dataset_names)
			grag_benchmark, grag_av_datasets = missing_benchmark_cloner(benchmark, num_tests, result_dir, Mode.GRAG, llm_name, corpus, retriever, k, dataset_names)

	if not parallel:
		datasets = {key: QADataset(key, benchmark) for key in av_datasets}
	else:
		cot_datasets = {key: QADataset(key, cot_benchmark) for key in cot_av_datasets}
		rag_datasets = {key: QADataset(key, rag_benchmark) for key in rag_av_datasets}
		grag_datasets = {key: QADataset(key, grag_benchmark) for key in grag_av_datasets}

	if parallel:
		mp.set_start_method("spawn", force=True)

		p_cot = Process(target=run_test, args=(Mode.COT, result_dir, cot_datasets, cot_av_datasets, llm_name, corpus, retriever, k, p4p))
		p_rag = Process(target=run_test, args=(Mode.RAG, result_dir, rag_datasets, rag_av_datasets, llm_name, corpus, retriever, k, p4p))
		p_grag = Process(target=run_test, args=(Mode.GRAG, result_dir, grag_datasets, grag_av_datasets, llm_name, corpus, retriever, k, p4p))

		p_cot.start()
		p_rag.start()
		p_grag.start()

		p_cot.join()
		p_rag.join()
		p_grag.join()

	else:
		run_test(Mode(mode), result_dir, datasets, av_datasets, llm_name, corpus, retriever, k, p4p)

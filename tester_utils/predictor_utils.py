from main_src.medgraphrag import Mode
import torch
import json
import gc


def get_mode(mode):
	if isinstance(mode, Mode):
		mode = mode.value
	elif isinstance(mode, str):
		mode = mode
	return mode

def get_dataset_names(json_path):
	with open(json_path, "r") as f:
		data = json.load(f)
	return list(data.keys())

def create_empty_folder_family(root, mode, llm_name, corpus, retriever_name, k, datasets):
	for dataset_name in datasets:
		if mode == Mode.GRAG:
			dirpath = root / dataset_name / "grag" / llm_name
		elif mode == Mode.RAG:
			dirpath = root / dataset_name / ("rag_" + str(k)) / llm_name / corpus / retriever_name
		else:
			dirpath = root / dataset_name / "cot" / llm_name
		dirpath.mkdir(parents=True, exist_ok=True)

def get_inner_folder_folder(res_folders, dataset, mode, llm_name, corpus, retriever_name, k):
	mode = get_mode(mode)
	
	if mode == "rag":
		method_dir = f"rag_{k}"
	else:
		method_dir = mode

	p = res_folders / dataset / method_dir / llm_name

	if mode == "rag":
		p = p / corpus / retriever_name

	return p

def count_prediction_files(folder):
	if not folder.exists():
		return 0
	return len([f for f in folder.iterdir() if f.is_file() and f.suffix == ".json"])

def missing_benchmark_cloner(original_benchmark, num_tests, res_folders, mode, llm_name, corpus, retriever_name, k, datasets):
	with open(original_benchmark, "r") as f:
		og_b = json.load(f)

	create_empty_folder_family(res_folders, mode, llm_name, corpus, retriever_name, k, datasets)

	new_benchmark = {}

	for dataset_dir in res_folders.iterdir():
		if not dataset_dir.is_dir():
			continue
		
		inner_folder = get_inner_folder_folder(res_folders, dataset_dir.name, mode, llm_name, corpus, retriever_name, k)
		cut_idx = count_prediction_files(inner_folder)

		dataset = dataset_dir.name

		if dataset not in og_b:
			continue

		questions = og_b[dataset]
		items = list(questions.items())

		# Se ho specificato lunghezza del dataset voglio arrivare massimo fino a quella risposta
		# Altrimenti vado fino in fondo
		if num_tests > 0:
			last_y = dict(items[cut_idx:num_tests])
		else:
			last_y = dict(items[cut_idx:])

		if last_y:
			new_benchmark[dataset] = last_y



	out_file = original_benchmark.with_name(get_mode(mode) + "_benchmark_" + str(num_tests) + ".json")
	with open(out_file, "w") as f:
		json.dump(new_benchmark, f, indent=4)

	return out_file, get_dataset_names(out_file)

def benchmark_cloner(original_benchmark, num_tests):
	if num_tests <= 0:
		return original_benchmark
	
	with open(original_benchmark, 'r') as f:
		data = json.load(f)

	new_benchmark = {}

	for section, questions in data.items():
		first_x = dict(list(questions.items())[:num_tests])
		new_benchmark[section] = first_x

	out_file = original_benchmark.with_name("benchmark_" + str(num_tests) + ".json")
	with open(out_file, 'w') as f:
		json.dump(new_benchmark, f, indent=4)

	return out_file

def test_and_write(model, mode, result_dir, datasets, av_dataset_names, llm_name, corpus, retriever_name, k, paths_for_pair):
	for dataset_name in av_dataset_names:
		dataset = datasets[dataset_name]
		
		if mode == Mode.GRAG:
			num_path_file = result_dir / dataset_name / (llm_name.split("/")[-1] + "_num_path_found_" + str(paths_for_pair) + ".txt")

		print("\n[{:s}]".format(dataset_name), end="\n")
		
		split = "test_"	
		if dataset_name == "medmcqa":
			split = "dev_"

		for q_idx in range(len(dataset)):
			if mode == Mode.GRAG:
				dirpath = result_dir / dataset_name / "grag" / llm_name
			elif mode == Mode.RAG:
				dirpath = result_dir / dataset_name / ("rag_" + str(k)) / llm_name / corpus / retriever_name
			else:
				dirpath = result_dir / dataset_name / "cot" / llm_name

			dirpath.mkdir(parents=True, exist_ok=True)

			qid = dataset.index[q_idx]
			filepath = dirpath / (split + str(qid) + ".json")

			question = dataset.dataset[dataset.index[q_idx]]["question"]
			options = dataset.dataset[dataset.index[q_idx]]["options"]
			
			print(mode.value + " - answering question: " + dataset.index[q_idx])

			if mode == Mode.GRAG:
				answer, num_path_found = model.answer(question=question, options=options, max_num_pairs=k, paths_for_pair=paths_for_pair)
				with open(num_path_file, "a") as f:
					line = split + dataset.index[q_idx] + ": " + str(num_path_found) + "\n"
					f.write(line)

			elif mode == Mode.RAG:
				answer, snippets, scores = model.answer(question=question, options=options, k=k)
			else:
				answer, _, _ = model.answer(question=question, options=options)
			answer = [answer]

			with open(filepath, "w") as f:
				json.dump(answer, f, indent=4)
	
	kill_model(model)


def kill_model(var):
	del var.model
	del var
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

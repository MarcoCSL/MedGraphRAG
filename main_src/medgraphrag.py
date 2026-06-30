from pathlib import Path
from enum import Enum
import json
import os
import re
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from openai_harmony import (
	load_harmony_encoding,
	HarmonyEncodingName
)
from main_src.utils import RetrievalSystem, DocExtracter
from main_src.template import *
from graph_src.GraphRAG import GraphRAG
from graph_src.graph_utils import *


PROJECT_DIR = Path(__file__).resolve().parent.parent

class Mode(Enum):
    COT = "cot"
    RAG = "rag"
    GRAG = "grag"

class MedGraphRAG:
	def __init__(self, llm_name="openai/gpt-oss-20b", mode=Mode.COT, retriever_name="MedCPT", corpus_name="Textbooks", db_dir=PROJECT_DIR/"corpus", cache_dir=None, corpus_cache=False, HNSW=False):
		self.device = "auto" if torch.cuda.is_available() else "cpu"
		self.llm_name = llm_name
		self.mode = mode
		self.retriever_name = retriever_name
		self.corpus_name = corpus_name
		self.db_dir = db_dir
		self.cache_dir = cache_dir
		self.docExt = None
		self.retrieve_definitions = False
		
		if self.mode == Mode.GRAG:
			self.retrieval_system = GraphRAG(self.device)
		elif self.mode == Mode.RAG:
				self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
		else:
			self.retrieval_system = None
			
		self.templates = {
			"cot_system": general_cot_system, "cot_prompt": general_cot,
			"medrag_system": general_medrag_system, "medrag_prompt": general_medrag,
			"graphrag_system": general_graphrag_system, "graphrag_prompt": general_graphrag
		}
		
		self.max_length = 2048
		self.context_length = 1024

		self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)

		if "mixtral" in llm_name.lower():
			self.tokenizer.chat_template = (PROJECT_DIR / "templates" / "mistral-instruct.jinja").read_text().replace('    ', '').replace('\n', '')
			self.max_length = 32768
			self.context_length = 30000
		elif "llama-2" in llm_name.lower():
			self.tokenizer.chat_template = (PROJECT_DIR / "templates" / "meditron.jinja").read_text().replace('    ', '').replace('\n', '')
			self.max_length = 4096
			self.context_length = 3072
		elif "llama-3" in llm_name.lower():
			self.tokenizer.chat_template = (PROJECT_DIR / "templates" / "meditron.jinja").read_text().replace('    ', '').replace('\n', '')
			self.max_length = 8192
			self.context_length = 7168
			if ".1" in llm_name or ".2" in llm_name:
				self.max_length = 1024
				self.context_length = 512
		elif "meditron-70b" in llm_name.lower():
			self.tokenizer.chat_template = (PROJECT_DIR / "templates" / "meditron.jinja").read_text().replace('    ', '').replace('\n', '')
			self.max_length = 4096
			self.context_length = 3072
			self.templates["cot_prompt"] = meditron_cot
			self.templates["medrag_prompt"] = meditron_medrag
		elif "pmc_llama" in llm_name.lower():
			self.tokenizer.chat_template = (PROJECT_DIR / "templates" / "pmc_llama.jinja").read_text().replace('    ', '').replace('\n', '')
			self.max_length = 2048
			self.context_length = 1024
		elif "gpt-oss-20b" in llm_name.lower():
			self.context_length = 128000
		
		if llm_name == "openai/gpt-oss-20b":
			if self.device == "cpu":
				print("Device must be set to cuda if you want to use openai/gpt-oss-20b!")
				print("Setting device to cuda!")
				self.device = "auto"
			self.encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
			self.model = AutoModelForCausalLM.from_pretrained(
				self.llm_name, 
				device_map=self.device
			)
		else:
			self.model = transformers.pipeline(
				"text-generation",
				model=self.llm_name,
				tokenizer=self.tokenizer,
				device_map=self.device,
				model_kwargs={
					"cache_dir": self.cache_dir,
					"torch_dtype": torch.bfloat16,
				},
			)
		
		self.answer = self.medgraphrag_answer

	def custom_stop(self, stop_str, input_len=0):
		stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
		return stopping_criteria

	def generate(self, messages, **kwargs):
		'''
		generate response given messages
		'''
		
		stopping_criteria = None
		prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

		# print(prompt)

		if "meditron" in self.llm_name.lower():
			# stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
			stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))

		if self.llm_name == "openai/gpt-oss-20b":
			query = messages[0]
			options = messages[1]
			question = "Answer the question:\n" + query + "\n\nPossible options:\n" + options
			
			if messages[2] != None:
				question = "Knowing this information:\n" + messages[2] + "\n\n" + question
			
			# print(question, "\n")
			ans = GPT_OSS_answer(self.model, self.encoder, question)
			return ans
		
		elif "llama-3" in self.llm_name.lower():
				response = self.model(
					prompt,
					do_sample=False,
					eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
					pad_token_id=self.tokenizer.eos_token_id,
					max_new_tokens=self.max_length,
					truncation=True,
					stopping_criteria=stopping_criteria,
					**kwargs
				)
		else:
			response = self.model(
				prompt,
				do_sample=False,
				eos_token_id=self.tokenizer.eos_token_id,
				pad_token_id=self.tokenizer.eos_token_id,
				max_new_tokens=self.max_length,
				truncation=True,
				stopping_criteria=stopping_criteria,
				**kwargs
			)

		torch.cuda.empty_cache()
		ans = response[0]["generated_text"][len(prompt):]
		return ans

	def medgraphrag_answer(self, question, options=None, max_num_pairs=10, paths_for_pair=2, retrieve_definitions=True, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
		'''
		question (str): question to be answered
		options (Dict[str, str]): options to be chosen from
		k (int): number of snippets to retrieve
		rrf_k (int): parameter for Reciprocal Rank Fusion
		save_dir (str): directory to save the results
		snippets (List[Dict]): list of snippets to be used
		snippets_ids (List[Dict]): list of snippet ids to be used
		'''

		if options is not None:
			options = self.options_formatter(options)
		else:
			options = ''

		# Retrieve relevant snippets
		if self.mode == Mode.GRAG:
			cleaned_options = "\n".join(
				line.split(": ", 1)[1]
				for line in options.splitlines()
			)
			contexts, num_path = self.retrieval_system.search(question, cleaned_options, max_num_pairs, paths_for_pair, retrieve_definitions)

			if contexts == "":
				contexts = [""]
			else:
				contexts = [self.tokenizer.decode(self.tokenizer.encode(contexts, add_special_tokens=False)[:self.context_length])]
		
		elif self.mode == Mode.RAG:
			if snippets is not None:
				retrieved_snippets = snippets[:k]
				scores = []
			elif snippets_ids is not None:
				if self.docExt is None:
					self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
				retrieved_snippets = self.docExt.extract(snippets_ids[:k])
				scores = []
			else:
				assert self.retrieval_system is not None
				retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

			contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
			if len(contexts) == 0:
				contexts = [""]
			else:
				contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
		
		else:
			retrieved_snippets = []
			scores = []
			contexts = None

		context = contexts[0] if contexts else None

		if save_dir is not None and not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		# Generate answers
		answers = []

		if self.mode == Mode.RAG:
			prompt_template = "medrag_prompt"
			prompt_system = "medrag_system"
		elif self.mode == Mode.GRAG:
			prompt_template = "graphrag_prompt"
			prompt_system = "graphrag_system"
		else:
			prompt_template = "cot_prompt"
			prompt_system = "cot_system"

		user_prompt = self.templates[prompt_template].render(context=context, question=question, options=options)			
		messages = [
			{"role": "system", "content": self.templates[prompt_system]},
			{"role": "user", "content": user_prompt}
		]
		
		if self.llm_name == "openai/gpt-oss-20b":
			messages = (question, options, context)
			ans = self.generate(messages, **kwargs)
		else:
			ans = self.generate(messages, **kwargs)
		answers.append(re.sub("\s+", " ", ans))
		
		if save_dir is not None:
			with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
				json.dump(retrieved_snippets, f, indent=4)
			with open(os.path.join(save_dir, "response.json"), 'w') as f:
				json.dump(answers, f, indent=4)
		
		if self.mode == Mode.GRAG and len(answers) == 1:
			return answers[0], num_path
		else:
			return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores
		
	def options_formatter(self, options):
		return "\n".join(f"{k}: {v}" for k, v in options.items())


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)

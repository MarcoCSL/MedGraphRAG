from itertools import combinations
import random

from graph_src.EntityLinker import EntityLinker
from graph_src.NER import NER
from graph_src.DBInterviewer import DBInterviewer


class GraphRAG():
	def __init__(self, device):
		self.ner = NER()
		self.entity_linker = EntityLinker(device)
		self.db_interviewer = DBInterviewer()

	def search(self, question, options, max_num_pairs=10, paths_for_pair=2, retrieve_definitions=True):
		query = self.input_formatter(question, options)
		try:
			entities = self.ner.find(query)

			linked_entities = set()
			for entity in entities:
				link_result = self.entity_linker.link(entity)

				if isinstance(link_result, list) and all(isinstance(r, list) for r in link_result):
					for r in link_result:
						linked_entities.add(tuple(r)) # If it has duplicates add them both, es:
						# [[127478, 'exposure', 'Phenol'], [17377, 'drug', 'Phenol']]
				else:
					linked_entities.add(tuple(link_result))

			linked_entities = list(linked_entities)
			pairs = list(combinations(linked_entities, 2))
			pairs = [(a, b) for (a, b) in pairs if a[2] != b[2]]

			if len(pairs) > max_num_pairs:
				pairs = random.sample(pairs, max_num_pairs)

			contexts = [self.db_interviewer.Yen(source, target, paths_for_pair) for source, target in pairs]
			context = self.context_cleaner(contexts, retrieve_definitions)

			return context, self.num_paths(contexts)

		except Exception as e:
			print(f"Error during search: {e}")
			raise
	
	def input_formatter(self, question, options):
		question += "\nOptions:" + options
		return question
	
	def context_cleaner(self, contexts, retrieve_definitions):
		all_definitions = []
		for path in contexts:
			for row in path:
				for name, definition in row[2]:
					all_definitions.append((name, definition))

		seen = set()
		unique_definitions = []
		for key, value in all_definitions:
			if key not in seen:
				seen.add(key)
				unique_definitions.append((key, value))

		cleaned_definitions = "\n\n\nDefinitions:\n"

		for name, info in unique_definitions:
			cleaned_definitions += f"\n{name}\n"
			for key, value in info.items():
				cleaned_definitions += f"{key}: {value}\n"

		paths = []
		for sublist in contexts:
			for _, path, _ in sublist:
				paths.append(path)

		final_context = "\n\n".join(f"Path {i+1}: {p}" for i, p in enumerate(paths))
		
		if retrieve_definitions:
			final_context += cleaned_definitions

		return final_context
	
	def num_paths(self, contexts):
		c = 0
		for sub in contexts:
			for _ in sub:
				c += 1
		return c
	

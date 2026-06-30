from pathlib import Path
import json
import re

class QADataset:
	def __init__(self, data, filename, dir="."):
		self.data = data.lower().split("_")[0]
		benchmark = json.load(open(Path(dir) / filename))
		if self.data not in benchmark:
			raise KeyError("{:s} not supported".format(data))
		self.dataset = benchmark[self.data]
		self.index = sorted(self.dataset.keys())
		
	# Numero domande nel sotto-dizionario (cioè num domande del dataset "dataset")
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, key):
		if type(key) == int:
			return self.dataset[self.index[key]]
		elif type(key) == slice:
			return [self.__getitem__(i) for i in range(self.__len__())[key]]
		else:
			raise KeyError("Key type not supported.")


def locate_answer(sentence:str):

	ans = re.findall("^\s*(A|B|C|D)$", sentence)
	if len(ans) > 0:
		return ans[0].upper()
	
	ans = re.findall("^\s*(A|B|C|D) or", sentence)
	if len(ans) > 0:
		return ans[0].upper()
	
	ans = re.findall("^\s*(A|B|C|D) and", sentence)
	if len(ans) > 0:
		return ans[0].upper()
		
	ans = re.findall("^\s*(A|B|C|D)/", sentence)
	if len(ans) > 0:
		return ans[0].upper()
	
	ans = re.findall("^\s*(A|B|C|D),", sentence)
	if len(ans) > 0:
		return ans[0].upper()
	
	ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
	if len(ans) > 0:
		return ans[0]

	ans = re.findall(":\s*(A|B|C|D)", sentence)
	if len(ans) > 0:
		return ans[0].upper()

	ans = re.findall("^\s*(A|B|C|D)\.", sentence)
	if len(ans) > 0:
		return ans[0].upper()

	ans = re.findall("^\s*(A|B|C|D)\"", sentence)
	if len(ans) > 0:
		return ans[0].upper()
	
	ans = re.findall("^\s*(A|B|C|D):", sentence)
	if len(ans) > 0:
		return ans[0].upper()

	return "A"

def locate_answer4pub_llama(sentence:str):

	sentence = sentence.split("Answer:")[-1]

	ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
	if len(ans) > 0:
		return ans[0]


	ans = re.findall("OPTION (A|B|C|D)", sentence)
	if len(ans) > 0:
		return ans[0]


	ans = re.findall("^\s*(A|B|C|D)\"", sentence)
	if len(ans) > 0:
		return ans[0].upper()
	
	ans = re.findall("^\s*(A|B|C|D):", sentence)
	if len(ans) > 0:
		return ans[0].upper()    

	return "A"
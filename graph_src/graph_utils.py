import torch
import re
from openai_harmony import (
	Conversation,
	Message,
	Role,
	SystemContent,
	DeveloperContent,
	ReasoningEffort
)


def GPT_OSS_answer(model, encoder, question):
	system = (
		SystemContent.new()
		.with_required_channels(["final"])
		.with_reasoning_effort(ReasoningEffort.MEDIUM)
	)

	developer_message = (
			DeveloperContent.new()
				.with_instructions(
					"You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options."
					"Always respond starting with 'Answer: ' followed by the correct option"
					"Do not include any explanation"
				)
		)

	convo = Conversation.from_messages([
		Message.from_role_and_content(Role.SYSTEM, system),
		Message.from_role_and_content(Role.DEVELOPER, developer_message),
		Message.from_role_and_content(Role.USER, question)
	])

	prefill_ids = encoder.render_conversation_for_completion(conversation=convo, next_turn_role=Role.ASSISTANT, config=None)
	stop_token_ids = encoder.stop_tokens_for_assistant_actions()

	input_ids_tensor = torch.tensor([prefill_ids], dtype=torch.long, device="cuda")

	outputs = model.generate(
		input_ids=input_ids_tensor,
		max_new_tokens=2048,#4096,
		eos_token_id=stop_token_ids
	)

	completion_ids = outputs[0][len(prefill_ids):]
	
	try:
		entries = encoder.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
		response = ""
		for message in entries:
			msg_dict = message.to_dict()
			if "final" in msg_dict['channel']:
				response = msg_dict['content'][0]['text']
				break
		
		ans = answer_extractor(response)
	except:
		raw_text = encoder.decode(completion_ids)
		ans = answer_extractor(raw_text)
	
	torch.cuda.empty_cache()

	return ans

def answer_extractor(text):
	text = text.lower().strip()
	
	# Cerca: "answer", "answer is", "[answer", "[answer is" seguiti da : o spazio e poi la risposta
	pattern = r"(?:\[\s*)?answer(?:\s+is)?[:\]]?\s*(a|b|c|d|yes|no|maybe)(?:\s*\]?)"
	
	match = re.search(pattern, text)
	if match:
		answer = match.group(1).strip()
		if len(answer) == 1:
			return answer.upper()
		else:
			return answer
	
	return text

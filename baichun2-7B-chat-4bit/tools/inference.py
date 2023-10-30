import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
# 路径是包含模型的文件夹路径，而非模型的完整路径
tokenizer = AutoTokenizer.from_pretrained("/volume/nlp/baichun2-7B-chat-4bit", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/volume/nlp/baichun2-7B-chat-4bit", device_map="cuda:1", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/volume/nlp/baichun2-7B-chat-4bit")
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)

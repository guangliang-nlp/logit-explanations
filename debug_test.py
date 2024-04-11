
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2Tokenizer, GPT2Model
name_path = "/scratch0/liuguan5/llama/llama-2-7b-chat-hf/"
tokenizer = LlamaTokenizer.from_pretrained(name_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = LlamaForCausalLM.from_pretrained(name_path, device_map='auto', output_attentions=True,
                                             return_dict=True)

for name, param in model.named_parameters():
    print(name)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
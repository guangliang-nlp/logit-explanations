from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AdamW,
    DataCollatorForLanguageModeling,
    AutoConfig
)

model = AutoModelForMaskedLM.from_pretrained("gpt2-xl")
for name, param in model.named_parameters():
    print(name)
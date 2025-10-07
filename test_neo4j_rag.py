from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
inputs = tokenizer("test", return_tensors="pt")
outputs = model(**inputs)
print("Output shape:", outputs.last_hidden_state.shape)

from transformers import AutoTokenizer, AutoModel
import torch



model_id = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

def embed(text: str, model: AutoModel.from_pretrained = model, tokenizer: AutoTokenizer.from_pretrained = tokenizer) -> list[float]:
    """
    Embedding text. Получаем dense_vector
    :param text:
    :param model:
    :param tokenizer:
    :return:
    """

    input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embedding = model(**input).last_hidden_state

    embedding = embedding.mean(dim=1).squeeze(0)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)  # l2 norm
    embedding = embedding.tolist()  # list[float]
    return embedding



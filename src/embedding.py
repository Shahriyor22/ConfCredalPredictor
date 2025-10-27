import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


conf_credal_predictor = "cross-encoder/nli-deberta-base"

def load_text_and_dist(jsonl_path):
    """Loads ChaosNLI examples from a JSONL file."""
    df = pd.read_json(jsonl_path, lines = True)

    prems = df["example"].apply(lambda x: x["premise"]).tolist()
    hypos = df["example"].apply(lambda x: x["hypothesis"]).tolist()
    dists = df["label_dist"].tolist()

    return prems, hypos, dists

def encode_batch(prems, hypos, batch_size = 16, device = "cpu"):
    """Encodes premise–hypothesis pairs with DeBERTa and returns CLS token embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(conf_credal_predictor)
    model = AutoModel.from_pretrained(conf_credal_predictor).to(device)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(prems), batch_size)):
        batch_prems = prems[i:i + batch_size]
        batch_hypos = hypos[i:i + batch_size]

        inputs = tokenizer(
            batch_prems, batch_hypos, return_tensors = "pt", padding = True, truncation = True, max_length = 256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings)

if __name__ == "__main__":
    prems, hypos, dists = load_text_and_dist("data/chaosNLI_mnli_m.jsonl")
    embeddings = encode_batch(prems, hypos, device = "mps")

    torch.save({
        "embeddings": embeddings,
        "distributions": torch.tensor(dists, dtype = torch.float32)
    }, "data/chaosNLI_mnli_m_embeddings.pt")

    print("Embeddings and distributions are saved.")

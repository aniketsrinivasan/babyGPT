from transformers import BigramLanguageModel
import torch


ROOT_DIR = "/Users/aniket/PycharmProjects/babyGPT"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reading input dataset into variable:
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# All the unique characters that occur in our data:
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoding input text into IDs:
#    creating a mapping:
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#    creating functions (lookup tables):
encode = lambda s: [stoi[c] for c in s]            # encoder: Str -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])   # decoder: list[int] -> Str


def main():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load(ROOT_DIR + "/trained_models/model_2.pt",
                                     map_location=device))
    model = model.to(device)

    context = torch.zeros([1, 1], dtype=torch.long, device=device)
    pred = model.generate(context, max_new_tokens=1000)[0].tolist()
    print(decode(pred))


if __name__ == "__main__":
    main()

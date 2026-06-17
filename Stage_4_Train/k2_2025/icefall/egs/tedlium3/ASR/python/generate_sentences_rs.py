import torch
import random
from icefall.transformer_lm.model import TransformerLM
from icefall.checkpoint import load_checkpoint
import sentencepiece as spm
import sys

# Usage: python generate_sentences.py <checkpoint> <bpe_model> [num_sentences] [temperature]
#
# temperature controls the randomness of the sampling:
#   - temperature = 1.0 (default): sample according to the model's learned probabilities
#   - temperature < 1.0 (e.g. 0.3): makes the distribution "sharper" — the highest-probability
#     token gets even higher probability, so the output is more deterministic / repetitive.
#     As temperature -> 0, this approaches pure argmax (greedy) decoding.
#   - temperature > 1.0 (e.g. 2.0): makes the distribution "flatter" — all tokens get more
#     similar probabilities, so the output becomes more random / creative / chaotic.
#     As temperature -> infinity, this approaches uniform random sampling.
#
# Examples:
#   python generate_sentences.py model.pt bpe.model 5 0.5   # more deterministic
#   python generate_sentences.py model.pt bpe.model 5 1.0   # normal sampling
#   python generate_sentences.py model.pt bpe.model 5 1.5   # more random/creative

checkpoint_path = sys.argv[1]
bpe_model_path = sys.argv[2]
num_sentences = int(sys.argv[3]) if len(sys.argv) > 3 else 5
temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

# Load BPE model
sp = spm.SentencePieceProcessor()
sp.load(bpe_model_path)
vocab_size = sp.get_piece_size()

bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
eos_id = sp.eos_id() if sp.eos_id() != -1 else 1

# Load model
model = TransformerLM(
    vocab_size=vocab_size,
    embedding_dim=768,
    d_model=768,
    dim_feedforward=2048,
    nhead=8,
    num_layers=4,
    tie_weights=True,
)
model.eval()
load_checkpoint(checkpoint_path, model)


def generate_sentence(max_len=200, sos_id=bos_id, eos_id=eos_id, temperature=1.0):
    tokens = [sos_id]
    #print(f"[DEBUG] Initial tokens: {tokens}, temperature={temperature}")
    for _ in range(max_len):
        safe_tokens = [t if 0 <= t < vocab_size else eos_id for t in tokens]
        if any(t != tokens[i] for i, t in enumerate(safe_tokens)):
            print(f"[ERROR] Invalid token(s) in input: {tokens} -> {safe_tokens}")
        x = torch.tensor([safe_tokens], dtype=torch.long)
        x_lens = torch.tensor([len(safe_tokens)], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(x, x, x_lens, return_logits=True)
            # Apply temperature: divide logits by temperature before softmax.
            # Lower temperature -> sharper distribution (more greedy)
            # Higher temperature -> flatter distribution (more random)
            scaled_logits = logits[0, -1] / temperature
            probs = torch.softmax(scaled_logits, dim=0)

            # Sort tokens by probability (descending) for cumulative sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # Random sampling via cumulative probability (inverse CDF):
            # Generate a random number r in [0, 1).
            # Walk through tokens sorted by probability (highest first),
            # accumulating their probabilities. Select the token where the
            # cumulative sum first exceeds r.
            r = random.random()
            cumulative = 0.0
            next_token = sorted_indices[0].item()  # fallback to argmax
            for i in range(len(sorted_probs)):
                cumulative += sorted_probs[i].item()
                if r < cumulative:
                    next_token = sorted_indices[i].item()
                    break

        if next_token < 0 or next_token >= vocab_size:
            print(f"[ERROR] Invalid token index: {next_token} (vocab_size={vocab_size})")
            next_token = eos_id
        if next_token == eos_id:
            break
        tokens.append(next_token)
    # Decode all tokens, then replace <sos/eos> with newline
    text = sp.decode(tokens[1:])
    return text.replace("<sos/eos>", "\n")


print(f"[INFO] vocab_size={vocab_size}, bos_id={bos_id}, eos_id={eos_id}, temperature={temperature}")
print("Generated sentences:")
for _ in range(num_sentences):
    sentence = generate_sentence(sos_id=bos_id, eos_id=eos_id, temperature=temperature)
    print(sentence)

import torch
from icefall.transformer_lm.model import TransformerLM
from icefall.checkpoint import load_checkpoint
import sentencepiece as spm
import sys

# Usage: python generate_sentences.py <checkpoint> <bpe_model> <num_sentences>
checkpoint_path = sys.argv[1]
bpe_model_path = sys.argv[2]
num_sentences = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# Load BPE model
sp = spm.SentencePieceProcessor()
sp.load(bpe_model_path)
vocab_size = sp.get_piece_size()


bos_id = sp.bos_id() if sp.bos_id() != -1 else 0
eos_id = sp.eos_id() if sp.eos_id() != -1 else 0

# Instantiate model and load checkpoint before function definition
model = TransformerLM(
    vocab_size=vocab_size,
    num_layers=4,
    tie_weights=True,
)
model.eval()
load_checkpoint(checkpoint_path, model)

def generate_sentence(max_len=30, sos_id=bos_id, eos_id=eos_id):
    tokens = [sos_id]
    print(f"[DEBUG] Initial tokens: {tokens}")
    for _ in range(max_len):
        safe_tokens = [t if 0 <= t < vocab_size else eos_id for t in tokens]
        if any(t != tokens[i] for i, t in enumerate(safe_tokens)):
            print(f"[ERROR] Invalid token(s) in input: {tokens} -> {safe_tokens}")
        x = torch.tensor([safe_tokens], dtype=torch.long)
        x_lens = torch.tensor([len(safe_tokens)], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(x, x, x_lens, return_logits=True)
            # Sample next token randomly from logits
            probs = torch.softmax(logits[0, -1], dim=0)
            next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token < 0 or next_token >= vocab_size:
            print(f"[ERROR] Invalid token index: {next_token} (vocab_size={vocab_size})")
            next_token = eos_id
        tokens.append(next_token)
        if next_token == eos_id:
            break
    return sp.decode(tokens[1:-1])

print(f"[INFO] vocab_size={vocab_size}, bos_id={bos_id}, eos_id={eos_id}")
print("Generated sentences:")
for _ in range(num_sentences):
    sentence = generate_sentence(sos_id=bos_id, eos_id=eos_id)
    print(sentence)
    
    # Load model
    model = TransformerLM(
        vocab_size=vocab_size,
    num_layers=4,
)
model.eval()
load_checkpoint(checkpoint_path, model)

def generate_sentence(max_len=30, sos_id=sp.bos_id(), eos_id=sp.eos_id()):
    tokens = [sos_id]
    print(f"[DEBUG] Initial tokens: {tokens}")
    for _ in range(max_len):
        # Clamp all tokens to valid range before passing to embedding
        safe_tokens = [t if 0 <= t < vocab_size else eos_id for t in tokens]
        if any(t != tokens[i] for i, t in enumerate(safe_tokens)):
            print(f"[ERROR] Invalid token(s) in input: {tokens} -> {safe_tokens}")
        x = torch.tensor([safe_tokens], dtype=torch.long)
        x_lens = torch.tensor([len(safe_tokens)], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(x, x, x_lens, return_logits=True)
            next_token = logits[0, -1].argmax().item()
        # Clamp next_token to valid range
        if next_token < 0 or next_token >= vocab_size:
            print(f"[ERROR] Invalid token index: {next_token} (vocab_size={vocab_size})")
            next_token = eos_id  # Force EOS to stop generation
        tokens.append(next_token)
        if next_token == eos_id:
            break
    return sp.decode(tokens[1:-1])  # Remove SOS/EOS for output

print("Generated sentences:")
for _ in range(num_sentences):
    print(generate_sentence(sos_id=bos_id, eos_id=eos_id))

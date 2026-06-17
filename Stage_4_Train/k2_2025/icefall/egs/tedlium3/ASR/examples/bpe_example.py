import sentencepiece as spm

# Load the trained model
sp = spm.SentencePieceProcessor()
sp.load("data/lang_bpe_500/bpe.model")

# Convert words to tokens
words = ["understanding", "hello", "world", "speech", "recognition"]

for word in words:
    pieces = sp.encode(word, out_type=str)    # subword strings
    ids = sp.encode(word, out_type=int)        # token IDs (0-499)
    print(f"{word:20s} → pieces: {pieces}  ids: {ids}")

# You can also encode full sentences
sentence = "this is a test sentence"
pieces = sp.encode(sentence, out_type=str)
ids = sp.encode(sentence, out_type=int)
print(f"\nSentence: '{sentence}'")
print(f"Pieces:   {pieces}")
print(f"IDs:      {ids}")

# And decode back from IDs to text
decoded = sp.decode(ids)
print(f"Decoded:  '{decoded}'")
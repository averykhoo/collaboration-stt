# pip install datasets soundfile tqdm

from datasets import load_dataset
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import sys

# 1️⃣ set your target folder
out_dir = Path(sys.argv[1])
out_dir.mkdir(exist_ok=True)

# 2️⃣ load Malay portion
# dataset = load_dataset("google/fleurs", "ms_my", split="train+validation+test")
dataset = load_dataset("google/fleurs", "ms_my", split="test", trust_remote_code=True)

# 3️⃣ export each sample
for i, example in enumerate(tqdm(dataset, desc="Exporting")):
    wav_path = out_dir / f"{i+1}.wav"
    txt_path = out_dir / f"{i+1}.txt"
    
    audio = example["audio"]
    sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(example["transcription"])

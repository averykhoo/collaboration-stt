#!/bin/bash
set -euo pipefail

wdir=/mount/data/alx/Stage_4_Train/k2_2025/icefall/egs/tedlium3/ASR
out_dir=/mount/data/alx/models/s0001
cd "$wdir"

export PYTHONPATH='/mount/data/alx/Stage_4_Train/k2_2025/icefall/egs/tedlium3/ASR/zipformer:/mount/data/alx/Stage_4_Train/k2_2025/icefall:$PYTHONPATH'
export CUDA_VISIBLE_DEVICES="0"

epoch=50
avg=20

# ---- Inject a temporary sitecustomize to allow-list PosixPath ----
PATCH_DIR="$(mktemp -d)"
cat > "$PATCH_DIR/sitecustomize.py" <<'PY'
import pathlib
try:
    import torch, torch.serialization
    # Allow-list classes used in your checkpoints
    torch.serialization.add_safe_globals([
        pathlib.PosixPath,
        pathlib.PurePosixPath,  # optional, harmless
    ])
except Exception:
    # keep startup resilient even if torch isn't available yet
    pass
PY
# Prepend so it loads first
export PYTHONPATH="$PATCH_DIR:$PYTHONPATH"
# ------------------------------------------------------------------

# Run export
PYTHONWARNINGS='ignore::FutureWarning' \
python ./zipformer/export.py \
  --exp-dir zipformer/exp \
  --epoch "${epoch}" \
  --avg "${avg}"

rm -fr ${out_dir}
mkdir ${out_dir}
cp -f zipformer/exp/pretrained.pt ${out_dir}
cp -f data/lang_bpe_500/bpe.model ${out_dir}
cp -f data/lang_bpe_500/tokens.txt ${out_dir}
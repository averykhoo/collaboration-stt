import torch

checkpoint = torch.load('/mount/data/alx/Stage_4_Train/k2_2025/icefall/egs/tedlium3/ASR/zipformer/exp/epoch-21.pt', map_location='cpu', weights_only=False)
state_dict = checkpoint['model']

encoder_params = 0
decoder_params = 0
joiner_params = 0
other_params = 0

for name, param in state_dict.items():
    numel = param.numel()
    if name.startswith('encoder.'):
        encoder_params += numel
    elif name.startswith('decoder.'):
        decoder_params += numel
    elif name.startswith('joiner.'):
        joiner_params += numel
    else:
        other_params += numel

total = encoder_params + decoder_params + joiner_params + other_params

print(f'Encoder:  {encoder_params:>12,} params  ({encoder_params/1e6:.2f}M)')
print(f'Decoder:  {decoder_params:>12,} params  ({decoder_params/1e6:.2f}M)')
print(f'Joiner:   {joiner_params:>12,} params  ({joiner_params/1e6:.2f}M)')
print(f'Other:    {other_params:>12,} params  ({other_params/1e6:.2f}M)')
print(f'Total:    {total:>12,} params  ({total/1e6:.2f}M)')
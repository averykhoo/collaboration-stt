"""Make far-field versions of a clean clip with FAST's FASTSynthesisTemplate.

Runs in the Farfield-Audio-Synthesis-Toolkit env, not this repo's:
    <fast-env>/python.exe noise_eval/augment.py CLEAN.wav OUT_DIR [--draws 5]

Grid: every (room-IR bucket, SNR) pair except (none, none), which is the clean clip.
  * RT60 bucket: FAST room-IR folder. A random real IR is picked from it per
    draw, and its measured T30 is logged. `none` means no room reverb.
  * SNR: a stationary-noise bed (random clip, random seeded slicing) mixed at
    that SNR against the (reverberant) speech. `none` means no noise.
All other stages are off: tempo 1.0 / pitch 0.0 (rubberband bypassed, so the
output stays time-aligned), no occlusion, device IR, codec or phone lowpass,
and no room IR on the noise.

Common random numbers: draw d uses the same noise clip and slice seed at every
SNR and bucket, and the same IR at every SNR within a bucket. Differences
between cells therefore reflect the RT60/SNR setting, not a different draw.

Writes OUT_DIR/<id>.wav (16-bit PCM, the FAST dirty output) and
OUT_DIR/manifest.csv (id, wav, rt60, snr, draw, rt60_measured, ir, noise, slice_seed).
Existing WAVs are kept, so the run is resumable.
"""
import argparse
import csv
import os
import random
import sys
import zlib

FAST = "C:/Users/user/PycharmProjects/Farfield-Audio-Synthesis-Toolkit"
DATA = FAST + "/data/input"
RT60_BUCKETS = ["none", "0.4s", "0.8s"]
SNRS_DB = ["none", 20, 10, 5, 0]


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clean_wav")
    p.add_argument("out_dir")
    p.add_argument("--draws", type=int, default=5)
    a = p.parse_args()

    clean = os.path.abspath(a.clean_wav).replace("\\", "/")
    out_dir = os.path.abspath(a.out_dir).replace("\\", "/")
    sys.path.insert(0, FAST)
    os.chdir(FAST)
    import soundfile as sf
    from src.workflow import ExecutionContext, ExecutionGraph, FASTSynthesisTemplate

    os.makedirs(out_dir, exist_ok=True)
    # Node paths must be relative to the context dirs, so root the context at
    # the common parent of both repos and the output.
    root = os.path.commonpath([clean, out_dir, DATA]).replace("\\", "/")
    rel = lambda x: os.path.relpath(x, root).replace("\\", "/")
    ctx = ExecutionContext(input_dir=root, output_dir=root)

    with open(f"{DATA}/Impulse_Responses/room_IRs/rir_metadata_audit.csv", newline="") as f:
        t30 = {r["file_name"]: float(r["actual_physical_rt60"]) for r in csv.DictReader(f)}
    irs = {b: sorted(x for x in os.listdir(f"{DATA}/Impulse_Responses/room_IRs/{b}") if x.endswith(".npy"))
           for b in RT60_BUCKETS if b != "none"}
    noises = sorted(os.listdir(f"{DATA}/01_stationary_noise"))

    rows = []
    for d in range(a.draws):
        nrng = random.Random(seed_of("noise", d))
        noise, slice_seed = nrng.choice(noises), nrng.randint(0, 2 ** 31 - 1)
        for b in RT60_BUCKETS:
            ir = random.Random(seed_of("ir", b, d)).choice(irs[b]) if b != "none" else None
            for snr in SNRS_DB:
                if b == "none" and snr == "none":
                    continue
                rid = f"rt{b.rstrip('s')}_snr{snr}_d{d}"
                entry = {
                    "original_speech_file": os.path.basename(clean),
                    "generate_clean_speech": {"tempo_change_rate": 1.0, "pitch_shift": 0.0},
                    "add_room_reverb": {"RIRs_used": [f"{b}/{ir}"]} if ir else None,
                    "stationary_noise": None if snr == "none" else {"noise_0": {
                        "noise_name": noise, "room_ir": None, "stationary_slice_seed": slice_seed}},
                    "nonstationary_noise": None,
                    "combine_speech_noise": {"speech_noise_SNR": 0.0 if snr == "none" else float(snr),
                                             "stationary_nonstationary_NNR": 0.0},
                    "simulate_occlusion": None, "simulate_device": None,
                    "simulate_codec": None, "phone_lowpass": None,
                }
                wav = f"{out_dir}/{rid}.wav"
                rows.append(dict(id=rid, wav=wav, rt60=0.0 if b == "none" else float(b.rstrip("s")),
                                 snr="inf" if snr == "none" else float(snr), draw=d,
                                 rt60_measured=t30[ir[:-4]] if ir else 0.0, ir=ir or "",
                                 noise=noise if snr != "none" else "", slice_seed=slice_seed if snr != "none" else ""))
                if os.path.exists(wav):
                    continue
                t = FASTSynthesisTemplate(
                    entry=entry, speech_dir=rel(os.path.dirname(clean)),
                    room_ir_dir=rel(f"{DATA}/Impulse_Responses/room_IRs"),
                    noise_stationary_dir=rel(f"{DATA}/01_stationary_noise"),
                    noise_nonstationary_dir=rel(f"{DATA}/02_non-stationary_noise"),
                    occlusion_ir_dir=rel(f"{DATA}/Impulse_Responses/occlusion_IRs"),
                    device_ir_dir=rel(f"{DATA}/Impulse_Responses/device_IRs"),
                    clean_out_path=rel(f"{out_dir}/unused_clean.wav"), dirty_out_path=rel(wav))
                ch = t._assemble()
                y = ExecutionGraph(leaves=[ch.dirty]).execute(ctx)[ch.dirty].waveform
                y = y.detach().cpu().numpy().reshape(-1)
                sf.write(wav + ".part.wav", y, 16000, subtype="PCM_16")
                os.replace(wav + ".part.wav", wav)
                print(rid, len(y), ir, noise if snr != "none" else "", flush=True)

    with open(f"{out_dir}/manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs in {out_dir}/manifest.csv")


if __name__ == "__main__":
    main()

import librosa
import numpy as np
import mir_eval

gt_vocal, sr_gt = librosa.load('data/vocal_trimmed.wav', sr=None)
gt_instr, sr_gt = librosa.load('data/inst_trimmed.wav', sr=None)
print(f"SR (GT): {sr_gt}")
print(f"Length vocal: {gt_vocal.shape}, Length instrumental: {gt_instr.shape}")

est_vocal, sr_est = librosa.load('results/vocal_est.wav', sr=None)
est_instr, sr_est = librosa.load('results/inst_est.wav', sr=None)
print(f"SR (estimated): {sr_est}")
print(f"Length estimated vocal: {est_vocal.shape}, Length estimated instrumental: {est_instr.shape}")

# resample if sample rates do not match
if sr_est != sr_gt:
    est_vocal = librosa.resample(est_vocal, orig_sr=sr_est, target_sr=sr_gt)
    est_instr = librosa.resample(est_instr, orig_sr=sr_est, target_sr=sr_gt)

min_len = min(len(gt_vocal), len(est_vocal))
gt_vocal, est_vocal = gt_vocal[:min_len], est_vocal[:min_len]
gt_instr, est_instr = gt_instr[:min_len], est_instr[:min_len]

print(f"After resampling and trimming:")
print(f"Length vocal: {gt_vocal.shape}, Length instrumental: {gt_instr.shape}")
print(f"Length estimated vocal: {est_vocal.shape}, Length estimated instrumental: {est_instr.shape}")


sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
    np.stack([gt_vocal, gt_instr], axis=0), 
    np.stack([est_vocal, est_instr], axis=0)
)

print("SDR:", sdr)
print("SIR:", sir)
print("SAR:", sar)
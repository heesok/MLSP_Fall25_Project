import librosa
import numpy as np
import mir_eval

def load_and_preprocess(gt_vocal_path, gt_inst_path, est_vocal_paths, est_inst_paths):
    """
    Args:
        gt_vocal_path (str): Path to ground truth vocal audio file.
        gt_inst_path (str): Path to ground truth instrumental audio file.
        est_vocal_paths (list[str]): List of estimated vocal file paths 
        est_inst_paths (list[str]): List of estimated instrumental file paths.

    Returns:
        tuple:
            - sr_common (int): Common sample rate.
            - gt_vocal (np.ndarray): Ground truth vocal waveform.
            - gt_inst (np.ndarray): Ground truth instrumental waveform.
            - est_vocals (list[np.ndarray]): List of aligned estimated vocals.
            - est_insts (list[np.ndarray]): List of aligned estimated instrumentals.
    """
    # Load ground truth
    gt_vocal, sr_gt = librosa.load(gt_vocal_path, sr=None)
    gt_inst, sr_gt2 = librosa.load(gt_inst_path, sr=None)
    assert sr_gt == sr_gt2, "Ground truth vocal and instrumental must have same sample rate"
    print(f"[GT] Sample rate: {sr_gt}, Vocal len: {len(gt_vocal)}, Inst len: {len(gt_inst)}")

    sr_common = sr_gt

    # Load estimated signals
    est_vocals = []
    est_insts = []
    for v_path, i_path in zip(est_vocal_paths, est_inst_paths):
        print(f'Loading estimated vocal: {v_path}, instrumental: {i_path} ...')
        est_v, sr_est_v = librosa.load(v_path, sr=None)
        est_i, sr_est_i = librosa.load(i_path, sr=None)

        # Resample to gt sample rate
        if sr_est_v != sr_common:
            est_v = librosa.resample(est_v, orig_sr=sr_est_v, target_sr=sr_common)
        if sr_est_i != sr_common:
            est_i = librosa.resample(est_i, orig_sr=sr_est_i, target_sr=sr_common)

        print(f"[Est] Sample rate: {sr_common}, Vocal len: {len(est_v)}, Inst len: {len(est_i)}")
        # Align lengths to the shortest one
        min_len = min(len(gt_vocal), len(est_v), len(gt_inst), len(est_i))
        est_v = est_v[:min_len]
        est_i = est_i[:min_len]

        est_vocals.append(est_v)
        est_insts.append(est_i)

    # Trim all
    min_len_all = min([len(gt_vocal)] + [len(e) for e in est_vocals]) 
    gt_vocal = gt_vocal[:min_len_all]
    gt_inst = gt_inst[:min_len_all]
    est_vocals = [e[:min_len_all] for e in est_vocals]
    est_insts = [e[:min_len_all] for e in est_insts]

    print(f"Common sample rate: {sr_common}, common length: {min_len_all}")
    return sr_common, gt_vocal, gt_inst, est_vocals, est_insts

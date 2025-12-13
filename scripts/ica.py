import os
import argparse
import librosa
import numpy as np
import soundfile as sf


def ica(X, n_components=None, eps=1e-12):
    # ICA function 
    M, N = X.shape
    if n_components is None:
        n_components = M
    print(f"ICA on data of shape {X.shape}, n_components={n_components}")

    # 1) Centering
    print('Centering ...')
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean

    # 2) Whitening
    print('Whitening ...')
    # covariance matrix, eigen-decomposition (Cov = E diag(λ) E^T)
    Cov = (Xc @ Xc.T) / N
    evals, E = np.linalg.eigh(Cov)

    # sort descending, avoid zeros
    idx = np.argsort(evals)[::-1] 
    evals, E = evals[idx], E[:, idx]
    evals = np.maximum(evals, eps) 

    # Keep top n_components
    E = E[:, :n_components]
    evals = evals[:n_components]

    # whitening
    print('  computing whitening matrix ...')
    C = np.diag(1.0 / np.sqrt(evals)) @ E.T 
    Z = C @ Xc # Whitened data (n x N)

    # 3) FOBI matrix D
    print('Computing FOBI matrix ...')
    r2 = np.sum(Z**2, axis=0) 
    D = (Z * r2) @ Z.T / N # N = number of cols

    # 4) Eigendecomposition of D (rotation in whitened space)
    d2, E2 = np.linalg.eigh(D)
    R = E2[:, np.argsort(d2)[::-1]] 

    # 5) Independent components
    H = R.T @ Z 

    return H

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/Tiki3_dev1/original.wav', help='Path to input mixture audio file')
    parser.add_argument('--output_dir', type=str, default='output/Tiki3/ICA', help='Directory to save output components')
    args = parser.parse_args()

    X, sr = librosa.load(args.input, mono=False) 
    print(f"Original shape: {X.shape}, Sample rate: {sr}")
    H = ica(X)
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(H.shape[0]):
        path = os.path.join(args.output_dir, f"comp{i}.wav")
        normalized_comp = H[i] / np.max(np.abs(H[i])) * 0.99 
        sf.write(path, normalized_comp, sr)
    print(f"Saved independent components to {args.output_dir}")

if __name__ == "__main__":
    main()
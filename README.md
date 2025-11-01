# MLSP F25 Final Project (Group 3)
- Hee So Kim (heesok@andrew.cmu.edu)
- Seo-Yoon Moon (smoon2@andrew.cmu.edu)

## File Structure
### Data
`data/`
- `die_with_a_smile.wav`: Original music file (Die with a smile)
- `die_with_a_smile_trimmed_51s.wav`: Trimmed original music of 51s
- `vocal_trimmed.wav`: Trimmed vocal-only track of the original music file (Downloaded from YouTube)
- `inst_trimmed.wav`: Trimmed instrument-only track of the original music file (Downloaded from YouTube)

### Codes
`scripts/`
- `NMF_BSS.ipynb`: Code for NMF blind source separation
- `ICA_BSS.ipynb`: Code for ICA blind source separation
- `evaluate.py`: Code for evaluating separated results compare to groud truth files
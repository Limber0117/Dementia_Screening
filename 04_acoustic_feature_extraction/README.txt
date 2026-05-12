
Clinical Audio Denoising Update
===============================

Files included:
1. clinical_denoise_pipeline.py
   - CPU-only denoising
   - Preserves pitch, jitter, pauses
   - Uses VAD-based noise estimation + gentle spectral reduction

2. ACOUSTIC_FEATURE_PATCH.txt
   - Minimal patch to integrate denoiser into AcousticFeatureExtractor

How to use:
-----------
1. Place clinical_denoise_pipeline.py in your project root
2. Import and call clinical_denoise_audio() inside preprocess_for_features()
3. Ensure config contains:
   - vad.mode (recommended: 2)
   - noise_reduction.aggressiveness (0.3–0.4)
   - noise_reduction.use_wiener = True

No silence is removed. Time axis is preserved.

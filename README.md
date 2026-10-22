# AI Project — Shot Classification

Double-click `run_demo.bat`. On its first run, it installs the required local packages and website packages; then your browser opens the demo.

1. Click **Browse files** and choose a short cricket broadcast clip containing one shot.
2. Click **Classify this shot**.
3. Show the prediction, top-five confidence chart, and the generated preprocessing preview.

The app uses the authors' four published weights in `models/`: player detector, bat detector, striker/bat segmenter, and the EfficientNetV2-S + GRU shot classifier. It does not upload videos anywhere; the frontend and model API both run on your computer.

For the clearest result, use a 1–5 second clip with a normal broadcast camera angle, only one intended batting shot, and the striker visible. Longer clips also work, but the model expects a single shot and samples 15 frames across the whole upload.

The reported class ordering is based on the dataset distribution order published in the repository README. The original training notebook generated label IDs from the extracted dataset-folder order; if you obtain the full dataset and see a different folder order, adjust `CLASS_NAMES` near the top of `app.py` to precisely match it.

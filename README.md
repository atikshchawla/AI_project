# AI Project — Shot Classification

Double-click `run_demo.bat`. On its first run, it installs the required local packages and website packages; then your browser opens the demo.

## Run it manually

Open two Git Bash terminals in this project folder. In the first terminal, start the API:

```bash
source .venv/Scripts/activate
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

In the second terminal, start the website:

```bash
cd frontend
npm run dev
```

Open the address shown by Vite (normally `http://localhost:5173`). The published classifier was saved with Keras 3.8. If you recreate the virtual environment, install the usual requirements first, then run `python -m pip install --force-reinstall --no-deps keras==3.8.0` before starting the API.

1. Click **Browse files** and choose a short cricket broadcast clip containing one shot.
2. Click **Classify this shot**.
3. Show the prediction, top-five confidence chart, and the generated preprocessing preview.

The app uses the authors' four published weights in `models/`: player detector, bat detector, striker/bat segmenter, and the EfficientNetV2-S + GRU shot classifier. It does not upload videos anywhere; the frontend and model API both run on your computer.

For the clearest result, use a 1–5 second clip with a normal broadcast camera angle, only one intended batting shot, and the striker visible. Longer clips also work, but the model expects a single shot and samples 15 frames across the whole upload.

The reported class ordering is based on the dataset distribution order published in the repository README. The original training notebook generated label IDs from the extracted dataset-folder order; if you obtain the full dataset and see a different folder order, adjust `CLASS_NAMES` near the top of `server.py` to precisely match it.

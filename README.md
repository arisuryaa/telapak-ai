# Telapak AI

A web app that registers a person's palm through a webcam and recognizes them when they scan it again.

> **Status:** the recognition model is being rebuilt. Until `backend/model.py` is filled in, Register
> and Scan stop after cutting the palm image and show "Model belum dibuat".

## How it works

1. **MediaPipe** finds the hand (21 points) and checks that the fingers are open and the palm faces the camera.
2. The center of the palm is cut into a 128×128 grayscale image (the ROI).
3. The **model** turns the ROI into a list of numbers (an embedding).
4. **Register** saves 8 embeddings per hand. **Scan** compares a new embedding with the saved ones
   using cosine distance. Photos are never stored.

## Project structure

```
backend/    Flask API (Python): model.py is where your model goes
frontend/   React + Vite + Tailwind web app
data/       datasets (not in git)
```

## Run it

Requirements: Python 3.10, Node.js 20+, MySQL 8 (Laragon).

```bash
# Backend: http://127.0.0.1:5000
python -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
cd backend
copy .env.example .env
..\.venv\Scripts\python app.py

# Frontend: http://localhost:5173 (in a second terminal)
cd frontend
npm install
npm run dev
```

## API docs

With the backend running, open **http://127.0.0.1:5000/api/docs**. The spec is in
[backend/openapi.yaml](backend/openapi.yaml).

## Adding a model

Edit [backend/model.py](backend/model.py):

```python
NAMA_MODEL = "my-model-v1"

def buat_embedding(roi):  # roi: 128x128 grayscale numpy array
    return [...]          # a list of numbers
```

Hands registered with a different model are not compared, so register again after changing models.

## Tests and formatting

```bash
cd backend && ..\.venv\Scripts\python -m pytest -q        # backend tests
cd backend && ..\.venv\Scripts\python -m black .          # format Python (Black)
cd frontend && npm test                                   # frontend tests
cd frontend && npm run format                             # format JS, CSS, YAML, Markdown (Prettier)
```

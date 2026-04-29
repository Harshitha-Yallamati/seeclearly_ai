# Deployment Guide — RetinoCheck

## Option 1: Docker (Recommended)

### Build & Run
```bash
docker-compose up --build
```
Access at `http://localhost:5001`

### With a Trained Model
Place your `dr_efficientnet_b3.keras` in the `models/` directory before building, or mount it:
```yaml
volumes:
  - ./models:/app/models
```

---

## Option 2: Render (Free Tier)

### Steps

1. **Push to GitHub** (ensure `models/` is in `.gitignore` to avoid large files)

2. **Create a Render Web Service**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repo
   - Build Command: `pip install -r requirements.txt && npm install && npm run build`
   - Start Command: `python backend/app.py`
   - Environment: Python 3.11

3. **Environment Variables**
   ```
   PORT=5001
   FLASK_ENV=production
   ```

4. **Upload Model**
   - Use Render's persistent disk or host the model on Google Drive/S3
   - Update `MODEL_PATH` in `backend/dr_detection.py` to point to the hosted model

### Limitations
- Free tier has 512MB RAM — TensorFlow may not fit
- Consider using TensorFlow Lite for smaller model footprint

---

## Option 3: HuggingFace Spaces (Best for ML demos)

### Steps

1. **Create a Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: Docker
   - Hardware: CPU Basic (free) or T4 Small (paid)

2. **Upload files**
   ```
   backend/
   models/
   utils/
   dist/        (built frontend: npm run build)
   Dockerfile
   requirements.txt
   ```

3. **Upload model weights**
   - HF Spaces allows up to 10GB of files
   - Upload `dr_efficientnet_b3.keras` directly

4. **Dockerfile** already handles everything — it builds the frontend and serves via Flask.

### Benefits
- Free GPU inference on T4 Small
- Auto-hosted with HTTPS
- Built-in model versioning

---

## Option 4: Local Development (Current Setup)

### Frontend Only (Mock Mode)
```bash
npm run dev
```
The UI works without a backend — predictions are simulated client-side.

### Full Stack
```bash
# Terminal 1
python backend/app.py

# Terminal 2
npm run dev
```

### With Docker Locally
```bash
docker-compose up --build
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5001` | Flask server port |
| `FLASK_ENV` | `development` | Set to `production` for deployment |
| `MODEL_PATH` | `models/dr_efficientnet_b3.keras` | Path to trained model |

---

## Common Issues

### "TensorFlow not found"
```bash
pip install tensorflow
```
On Apple Silicon: `pip install tensorflow-macos tensorflow-metal`

### "Port 5001 already in use"
```bash
# Kill the process using port 5001
# Windows:
netstat -ano | findstr :5001
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :5001
kill -9 <PID>
```

### "Model file too large for Git"
Add to `.gitignore`:
```
models/*.keras
models/*.h5
```
Host the model on Google Drive, S3, or HuggingFace Hub instead.

### "Out of Memory during training"
- Reduce `BATCH_SIZE` in `training/config.py` (try 16 or 8)
- Use mixed precision: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`

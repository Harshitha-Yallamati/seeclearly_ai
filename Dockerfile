# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Python Backend + Serve Frontend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY models/ ./models/
COPY utils/ ./utils/
COPY training/ ./training/

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/dist ./dist

# Expose port
EXPOSE 5001

# Run Flask server
CMD ["python", "backend/app.py"]

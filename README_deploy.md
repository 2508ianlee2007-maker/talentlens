# TalentLens — Docker Deployment Guide

## Your project structure

```
talentlens/
├── app.py
├── backend.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README_deploy.md   ← this file
```

---

## Option A — Run locally with Docker

> Best for testing before you push to the cloud.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Steps

```bash
# 1. Make sure all 6 files above are in the same folder
# 2. Open a terminal in that folder and build the image
docker compose up --build

# 3. Open your browser at:
#    http://localhost:8501
```

To stop: press `Ctrl+C`, then run `docker compose down`.

---

## Option B — Deploy on Render (free tier, recommended for school)

Render can deploy a Docker container directly from your GitHub repo for free.

### Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add app.py backend.py requirements.txt Dockerfile docker-compose.yml .dockerignore
   git commit -m "Initial commit"
   # Create a new repo on github.com, then:
   git remote add origin https://github.com/YOUR_USERNAME/talentlens.git
   git push -u origin main
   ```

2. **Create a Render account** at https://render.com (free, no credit card needed)

3. **New Web Service → Connect your GitHub repo**

4. **Configure the service:**
   | Setting | Value |
   |---|---|
   | Environment | Docker |
   | Region | Singapore (Southeast Asia) |
   | Branch | main |
   | Port | 8501 |
   | Plan | Free |

5. **Click Deploy** — Render will build your Docker image and give you a public URL like `https://talentlens.onrender.com`

> ⚠️ Free tier spins down after 15 min of inactivity. First load after sleep takes ~30 seconds.

---

## Option C — Deploy on Railway (free trial)

Railway gives you $5 free credit per month.

1. Go to https://railway.app and sign up with GitHub
2. Click **New Project → Deploy from GitHub repo**
3. Select your repo
4. Railway auto-detects the Dockerfile — no extra config needed
5. Under **Settings → Networking**, set the port to `8501`
6. Click **Deploy** — you'll get a public URL instantly

---

## Environment Variables (optional)

If you want to pre-bake your Groq API key instead of typing it in the UI each time, set this environment variable in Render/Railway:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_your_actual_key_here` |

Then add this one line at the top of `backend.py` (after the imports):
```python
import os
# Read key from env if not passed directly
_ENV_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
```

And in `app.py`, change the default value of `groq_key` in `_defaults` to:
```python
"groq_key": os.getenv("GROQ_API_KEY", ""),
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails on `faiss-cpu` | Make sure you're on `python:3.11-slim`, not Alpine |
| App crashes with `torch` error | The base image needs `libglib2.0-0` — already in the Dockerfile |
| 429 errors from Groq | Increase the **Delay between calls** slider in the UI to 2–3s |
| Render spins down | Upgrade to Render Starter ($7/mo) for always-on |
| Port not reachable | Confirm the platform's port setting is `8501` |

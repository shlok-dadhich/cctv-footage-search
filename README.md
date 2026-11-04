# CCTV Footage Search

Natural-language and reference-image search over indexed CCTV footage.

The app filters text queries to the requested entity (`person`, `vehicle`, or
`object`), shows the top 7 results, and lets you reveal more. Each result includes
a saved detection snapshot and, when available, a pre-cut footage clip.

## Demo data

This repository is a demo bundle. It ships:

- `outputs/cctv.db`: pre-built SQLite index with 768-d SigLIP embeddings
- `outputs/crops/`: saved detection snapshots
- `outputs/clips/`: pre-cut, 360p browser-friendly clips

Original source videos are not included because they are several gigabytes. The
app falls back to saved snapshots and pre-cut clips when a source video is absent.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On Linux/macOS use `source .venv/bin/activate` instead.

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** and select this repository, branch `main`, and `app.py`.
4. In **Advanced settings**, choose Python 3.12.
5. Deploy.

`packages.txt` installs `ffmpeg`, `libgl1`, and `libglib2.0-0` on the build
machine. The first text search downloads the SigLIP 2 model; image search also
downloads the person/vehicle detectors. Expect the first query to be slow on CPU.

## Notes

- The demo index contains public research footage (VIRAT and MEVA).
- SQLite is read-only at runtime for this demo. For a production pipeline, use
  Postgres with pgvector and object storage for videos and clips.
- `Dockerfile` is included if you later want to host the app on a container
  platform with a persistent disk.

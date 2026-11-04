# Person analytics pipeline

This folder provides GPU-first person detection and tracking without inferring
sensitive identity attributes. It outputs track IDs, face presence, heuristic
pose/orientation/activity, optional crops, and optional user-provided labels.

## Run

```powershell
python .\person\person_pipeline.py `
  --source "data\raw\New folder\cam1.mp4" `
  --output "person\output\cam1_analytics.mp4" `
  --device cuda:0 --half --track
```

`--track` is not required; tracking is always enabled by the pipeline.

Export crops:

```powershell
python .\person\person_pipeline.py `
  --source "data\raw\New folder\cam1.mp4" `
  --output "person\output\cam1_analytics.mp4" `
  --device cuda:0 --half `
  --crops "person\output\crops" --crop-every 15
```

Consent-based metadata overlay:

```json
{"1": "employee-A", "2": "visitor-consented"}
```

Save that as `person\metadata.json` and pass `--metadata-json person\metadata.json`.
The values are displayed only for matching tracker IDs. Face presence is not
face recognition. Pose, orientation, and activity are heuristic labels and
should not be treated as definitive.

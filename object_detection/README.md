# Object Detection

This folder contains the clothing and general object detection script and its generated videos.

Run the first five minutes of a video from the project root:

```powershell
.\.venv\Scripts\python.exe .\object_detection\detect_clothes_objects.py `
  --source "data\raw\New folder\cam1.mp4" `
  --output ".\object_detection\objects_out_first5min.mp4" `
  --device cuda `
  --half `
  --track `
  --max-duration 300
```

`--max-duration 300` limits processing to 300 seconds (5 minutes).

### Overview
This repository contains an implementation of a real-time pipeline that leverages vision and speech language models to identify and convey non-verbal social cues (such as speaker identification and nodding) to blind remote workers. This specific implementation uses MLX to allow for Apple GPU access.

The pipeline processes meeting recordings to:

1. Transcribe speech using Whisper.
2. Identify speakers based on name labels in video frames.
3. Assign transcript segments to speakers using a Qwen2.5-VL 7B (easily swapped out by changing the MLX model name)
4. Detect non-verbal cues like nodding in response to the speaker.

### Dependencies and Usage
To use, clone this repository and install the required dependencies:
```bash
pip install mlx-vlm==0.21.5 faster-whisper==1.1.1 opencv-python==4.10.0.84 pandas==2.2.3
```

Then, add a video to the directory. Then, run the following in terminal: 
```bash
python Social_Cue_Identification.py file_path file_name --model_name model_name_value
```

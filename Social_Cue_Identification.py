import glob
import os
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import pandas as pd
import json
from faster_whisper import WhisperModel
import cv2
import time

start_time = time.time()

# Helper functions
def predict_with_model(model, processor, image_url, formatted_prompt):
    output = generate(
        model, processor, formatted_prompt, image_url, 
        verbose=False, max_tokens=50,
        temperature=0.7, cache_history=None)
    return output


def setup():
    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config

def transcribe(filename):
    model_size = "small"

    # Run on GPU with Int 8
    model = WhisperModel(model_size, compute_type="int8")

    segments, info = model.transcribe(filename, beam_size=3)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    segments = list(segments)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return segments 

def identify_speakers(model, processor, image_url, config):
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the names of the people in the online meeting from their name labels. Be concise. Only give me the names and return them as strings in a list."}
    ]
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    output = generate(
        model, processor, formatted_prompt, [image_url], 
        verbose=False, max_tokens=70,
        temperature=0.7, cache_history=None)
    # process output into a list
    list_speakers = output.replace("'", "").replace("[", "").replace("]", "").split(", ")
    return list_speakers

def save_segment_video(filepath, video_name, segments):
    cap = cv2.VideoCapture(str(filepath))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    success, frame = cap.read()
    for i, segment in enumerate(segments):
        if not success:
            print('unsuccess')
            break
        if i==0:
            os.makedirs(f"segmented_videos/{video_name}", exist_ok=True)
            frame_path = os.path.join(f"segmented_videos/{video_name}/", f"frame_0.jpg")
            write = cv2.imwrite(frame_path, frame)
        os.makedirs(f"segmented_videos/{video_name}/segment_{i}", exist_ok=True)
        while frame_count/fps < segment.end:
            if frame_count % 25 == 0:
                frame_path = os.path.join(f"segmented_videos/{video_name}/segment_{i}/", f"frame_{frame_count}.jpg")
                write = cv2.imwrite(frame_path, cv2.resize(frame, (756, 448)))
                if write is False:
                    print(f"Error writing frame {frame_count} to {frame_path}")

            frame_count += 1
            success, frame = cap.read()

def assign_transcript_to_speakers(video_name, speakers, segments, model, processor, config):
    segment_dict = {}

    for i, segment in enumerate(segments):
        segment_images = glob.glob(f"segmented_videos/{video_name}/segment_{i}/*.jpg") # need list of images in the segment
        # obtain using filepath and segment starts and ends.
        prompt = [{"role": "system", "content":"You are a helpful assistant for a blind worker."},
            {"role": "user", "content": f"In my meeting recording, who is currently speaking? Help me identify who is saying the following script: {segment.text}. Be concise and only return a meeting member from: {speakers}."}
        ] 
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images = len(segment_images))
        output = generate(
        model, processor, formatted_prompt, segment_images, 
        verbose=False, max_tokens=50,
        temperature=0.7, cache_history=None)
        print(output)

        if output in speakers:
        # if output.lower() in [i.lower() for i in speakers]:
            segment_dict[i] = output
        else:
            print(f"speaker identified as {output}, which is not in the list of speakers: {speakers} \n")
            segment_dict[i] = "Unknown"
    print(segment_dict)
    return segment_dict

def identify_nodding(video_name, speakers, segment_dict, model, processor, config):
    nodding_dict = {}
    for i, speaker in segment_dict.items():
        print(f"\nSegment {i} with {speaker} speaking")
        nodding_dict[i] = []
        segment_images = glob.glob(f"segmented_videos/{video_name}/segment_{i}/*.jpg")
        for other_person in speakers:
            if other_person is not speaker:
                prompt = [
                    {"role": "system", "content": "You are a helpful assistant for a blind worker."},
                    {"role": "user", 
                        "content": [
                            {"type": "text", "text": f"Does {other_person} nod in the following video? Watch their head and see if it bobs in affirmation of the speaker. Answer True or False and explain why."},
                        ]
                    }
                ]
                formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(segment_images)) #len(segment_images))

                output = generate(
                    model, processor, formatted_prompt, segment_images, 
                    verbose=False, max_tokens=70,
                    temperature=0.7, cache_history=None)
                print(f"Segment {i}: {other_person} returns {output}")
                nodding_dict[i].append(other_person)
    print(nodding_dict)
    return nodding_dict





# Main setup
video_filepath = "Trimmed Code Review.mp4"
video_name = "Trimmed_Code_Review"

# Main Inference Loop
# 1. transcribing
transcript_segments = transcribe(video_filepath)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Transcript time: {elapsed_time} seconds")

# Splice video into screenshots from segments
save_segment_video(video_filepath, video_name, transcript_segments)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Segment time: {elapsed_time} seconds")

model, processor, config = setup()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Setup time: {elapsed_time} seconds")

# 2. identify speakers
image_url = f"segmented_videos/{video_name}/frame_0.jpg"
speakers = identify_speakers(model, processor, image_url, config)
print(f"Speakers are: {speakers}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time to calculate Speakers: {elapsed_time} seconds")

# 3. assign transcript to speakers
assigned_transcript_dict = assign_transcript_to_speakers(video_filepath, speakers, transcript_segments, model, processor, config)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time to assign speakers to each segment: {elapsed_time} seconds")

# identify nodding and cues based on (2, 3) with the video
identify_nodding(video_name, speakers, assigned_transcript_dict, model, processor, config)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time for identification of nodding: {elapsed_time} seconds")


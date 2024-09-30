import cv2
import subprocess
import openai
import numpy as np
import json
import math
import pickle
from tqdm import tqdm
import ffmpeg
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from youtube_transcript_api import YouTubeTranscriptApi

openai.api_key = '' # Use your api key here
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


# Segment Video function
def segment_video(response):
    for i, segment in enumerate(response):
        start_time = math.floor(float(segment.get("start_time", 0)))
        end_time = math.ceil(float(segment.get("end_time", 0)))
        output_file = f"output{str(i).zfill(3)}.mp4"

        # Adjust the start time slightly to avoid blank frames
        adjusted_start_time = max(0, start_time - 1)

        duration = end_time - adjusted_start_time
        command = f"ffmpeg -ss {adjusted_start_time} -i input_video.mp4 -t {duration} -c:v libx264 -c:a aac -strict experimental -b:a 192k {output_file}"
        subprocess.call(command, shell=True)

def detect_faces_dnn(frame):
    # Define image size for the DNN (300x300 is expected by the model)
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the blob as input to the network
    face_net.setInput(blob)

    # Perform forward pass to get the detections
    detections = face_net.forward()

    # Initialize a list to store the detected faces
    faces = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by setting a confidence threshold
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x2, y2) = box.astype("int")

            # Append the face bounding box (x, y, width, height)
            faces.append((x, y, x2 - x, y2 - y))

    return faces

def extract_audio(input_file, audio_file):
    """Extract audio from video using ffmpeg-python."""
    ffmpeg.input(input_file).output(audio_file).run(overwrite_output=True)

def merge_audio_video(video_file, audio_file, output_file, fps):
    """Merge audio and video using ffmpeg-python, preserving the frame rate."""
    # Ensure the streams are correctly set
    video_stream = ffmpeg.input(video_file, r=fps)
    audio_stream = ffmpeg.input(audio_file)
    ffmpeg.output(video_stream, audio_stream, output_file, vcodec='copy', acodec='aac').run(overwrite_output=True)

def detect_speaker_with_audio(input_file, output_file):
    try:
        # Temporary files for audio and video processing
        temp_audio_file = "temp_audio.aac"
        temp_video_file = "temp_video.mp4"

        # Extract audio from the original video
        extract_audio(input_file, temp_audio_file)

        # Constants for cropping
        CROP_RATIO = 0.9
        VERTICAL_RATIO = 9 / 16

        # Read the input video
        cap = cv2.VideoCapture(input_file)

        # Load Haar cascades for face and mouth detection
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        mouth_cascade = cv2.CascadeClassifier('.\haarcascade_mcs_mouth.xml')

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 360
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 640
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the original video
        # fps = 26

        target_height = int(frame_height * CROP_RATIO)  #324
        target_width = int(target_height * VERTICAL_RATIO)  #182

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(temp_video_file, fourcc, fps, (target_width, target_height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        last_speaker = None

        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
            detected_faces = detect_faces_dnn(frame)

            if len(detected_faces) > 0:
                speaker_candidate = None

                for face in detected_faces:
                    x, y, w, h = face
                    roi_gray = gray[y:y + h, x:x + w]

                    detected_mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

                    if len(detected_mouths) > 0:
                        speaker_candidate = face
                        break

                if speaker_candidate is not None:
                    last_speaker = speaker_candidate
                elif last_speaker is not None:
                    speaker_candidate = last_speaker

                if speaker_candidate is not None:
                    x, y, w, h = speaker_candidate

                    crop_x = max(0, x + (w - target_width) // 2)
                    crop_y = max(0, y + (h - target_height) // 2)
                    crop_x2 = min(crop_x + target_width, frame_width)
                    crop_y2 = min(crop_y + target_height, frame_height)

                    cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]
                    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

                    output_video.write(resized_frame)
                else:
                    crop_x = (frame_width - target_width) // 2
                    crop_y = (frame_height - target_height) // 2
                    crop_x2 = (frame_width + target_width) // 2
                    crop_y2 = (frame_height + target_height) // 2

                    cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]
                    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))
                    output_video.write(resized_frame)
            else:
                crop_x = (frame_width - target_width) // 2
                crop_y = (frame_height - target_height) // 2
                crop_x2 = (frame_width + target_width) // 2
                crop_y2 = (frame_height + target_height) // 2

                cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]
                resized_frame = cv2.resize(cropped_frame, (target_width, target_height))
                output_video.write(resized_frame)

        cap.release()
        output_video.release()

        # Merge the cropped video with the original audio, ensuring the frame rate is consistent
        merge_audio_video(temp_video_file, temp_audio_file, output_file, fps)

        # Clean up temporary files
        os.remove(temp_audio_file)
        os.remove(temp_video_file)

        print("Video cropped with audio successfully.")
    except Exception as e:
        print(f"Error during video cropping with audio: {str(e)}")

def get_transcript(video_id):
    # Get the transcript for the given YouTube video ID
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Format the transcript for feeding into GPT-4
    formatted_transcript = ''
    for entry in transcript:
        start_time = "{:.2f}".format(entry['start'])
        end_time = "{:.2f}".format(entry['start'] + entry['duration'])
        text = entry['text']
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

    return transcript

#Analyze transcript with GPT-4o-mini function
response_obj='''[
  {
    "start_time": 97.19, 
    "end_time": 127.43,
    "description": "Spoken Text here"
    "duration":36 #Length in seconds
  },
  {
    "start_time": 169.58,
    "end_time": 199.10,
    "description": "Spoken Text here"
    "duration":33 
  },
]'''

def analyze_transcript(transcript):
    prompt = f"This is a transcript of a video. Please identify the 3 most viral sections from the whole, make sure they are nearly 30 seconds in duration,Make Sure you provide extremely accurate timestamps respond only in this format {response_obj}  \n Here is the Transcription:\n{transcript}"
    messages = [
        {"role": "system",
         "content": "You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content"},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512,
        n=1,
        stop=None
    )
    return response.choices[0]['message']


if __name__=='__main__':
    video_id = 'iTwZzUApGkA'

    transcript = get_transcript(video_id)
    print(transcript)
    interesting_segment = analyze_transcript(transcript)
    print(interesting_segment)
    content = interesting_segment["content"]
    parsed_content = json.loads(content)
    print(parsed_content)

    # Save the parsed content if you want
    with open('parsed_content.pkl', 'wb') as file:
        pickle.dump(parsed_content, file)
    with open('parsed_content.pkl', 'rb') as file:
        parsed_content = pickle.load(file)

    segment_video(parsed_content)

    # # Loop through each segment
    for i in range(0, 3):  # Replace 3 with the actual number of segments
        input_file = f'output{str(i).zfill(3)}.mp4'
        output_file = f'output_cropped{str(i).zfill(3)}.mp4'
        detect_speaker_with_audio(input_file, output_file)

    video_paths = [f'output_cropped{str(i).zfill(3)}.mp4' for i in range(3)]
    video_clips = [VideoFileClip(video) for video in video_paths]
    final_clip = concatenate_videoclips(video_clips)

    # Write the output video file (with audio)
    final_clip.write_videofile('output_video.mp4', codec='libx264', audio_codec='aac')
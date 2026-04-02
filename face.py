import time

import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fer.fer import FER

"""
Real-Time Emotion Detection and Visualization

This script captures video from a webcam, detects emotions on faces in real-time,
and visualizes the results both in a live bar chart and in the video itself. It also
saves the video feed with detected emotions, the live bar chart as a GIF, and
cumulative emotion statistics over time as a static chart.
"""

# Set the backend for matplotlib to 'TkAgg' for compatibility with different environments
matplotlib.use("TkAgg")

# Initialize the FER (Face Emotion Recognition) detector using MTCNN
detector = FER(mtcnn=True)

# Start capturing video from the webcam (device 0)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Set a frame rate for recording the video
frame_rate = 4.3

# Initialize OpenCV's VideoWriter to save the video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("emotion_video.avi", fourcc, frame_rate, (640, 480))

# Set up a matplotlib figure for displaying live emotion detection results
plt.ion()
fig, ax = plt.subplots()
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ax.bar(emotion_labels, [0] * len(emotion_labels), color="lightblue")
ax.set_ylim(0, 1)
ax.set_ylabel("Confidence")
ax.set_title("Real-time Emotion Detection")
ax.tick_params(axis="x", labelrotation=45)
fig.tight_layout()

# Initialize imageio writer to save live chart updates as a GIF
gif_writer = imageio.get_writer("emotion_chart.gif", mode="I", duration=0.1)

# List to store cumulative emotion statistics for each frame
emotion_statistics = []


def update_chart(detected_emotions, ax, fig):
    ax.clear()
    ax.bar(
        emotion_labels,
        [detected_emotions.get(emotion, 0) for emotion in emotion_labels],
        color="lightblue",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Real-time Emotion Detection")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


# Start the timer to measure the active time of the webcam
webcam_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions on the frame
        result = detector.detect_emotions(frame)
        largest_face = None
        max_area = 0

        # Find the largest face in the frame for primary emotion analysis
        for face in result:
            x, y, w, h = face["box"]
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = face

        # If a face is detected, display the emotion and update the chart
        if largest_face:
            box = largest_face["box"]
            current_emotions = largest_face["emotions"]

            # Store the emotion data
            emotion_statistics.append(current_emotions)

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            emotion_type = max(current_emotions, key=current_emotions.get)
            emotion_score = current_emotions[emotion_type]
            emotion_text = f"{emotion_type}: {emotion_score:.2f}"
            cv2.putText(
                frame,
                emotion_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            update_chart(current_emotions, ax, fig)

            # Write the frame to the video file
            out.write(frame)

            # Save the current state of the bar chart as a frame in the GIF
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
            gif_writer.append_data(image)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    webcam_end_time = time.time()
    print(f"Webcam active time: {webcam_end_time - webcam_start_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)

    out.release()
    gif_writer.close()

    if emotion_statistics:
        emotion_df = pd.DataFrame(emotion_statistics)

        plt.figure(figsize=(10, 10))
        for emotion in emotion_labels:
            plt.plot(emotion_df[emotion].cumsum(), label=emotion)
        plt.title("Cumulative Emotion Statistics Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Cumulative Confidence")
        plt.legend()
        plt.savefig("cumulative_emotions.jpg")
        plt.close()
    else:
        print("No faces detected; skipped cumulative emotion chart.")
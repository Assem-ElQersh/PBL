import tensorflow as tf
from app import DataLoader
import numpy as np
import cv2, os, imageio
import warnings
warnings.filterwarnings("ignore")

model = tf.keras.models.load_model(os.path.join("C:", os.sep, "Users", "zi3dt", "pbl", "app", "conv1_lstm.keras"))


def predict(x):
    prediction = model.predict(x)
    predicted_label = DataLoader.arabic_labels[np.argmax(prediction, axis=1).item()]
    return predicted_label

def load_video(path):
    cap = cv2.VideoCapture(path)

    frames = []

    # Loop until the end of the video
    while(cap.isOpened()):
        # Read the frame
        ret, frame = cap.read()

        # If the frame was read correctly ret is True
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)

    # Release the VideoCapture object
    cap.release()

    # Convert the list of frames to a numpy array
    frames = np.array(frames)

    return frames

def main():
    file = os.path.join("C:", os.sep, "Users", "zi3dt", "pbl", "media", "uploaded_videos", "video.mp4") 
    # x = DataLoader.DataLoader.load_inference_data(file)
    x = DataLoader.DataLoader.load_inference_data(file)
    print(x.shape)
    prediction = predict(x)
    print(prediction)

if __name__ == "__main__":
    main()
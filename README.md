# Sign-to-Text Translation System Using Arabic Sign Language 

---

## Project Overview  

This project is a web application designed to translate Arabic sign language into text in real-time. It aims to bridge the communication gap between deaf and non-deaf individuals, promoting social inclusion and accessibility.

## Team and Supervision

- **Team Members:** 
- **Supervisor:** Dr. Ahmed Fares
- **Course:** Project-Based Learning (PBL)

## Problem Statement

Deaf individuals face numerous challenges, including communication barriers, limited access to education and employment, social exclusion, and difficulty accessing public services. Our project addresses these issues by providing a tool that translates Arabic sign language into text, facilitating better communication and inclusion.

## Project Objectives

1. **Enhance Communication:** Facilitate communication between sign language users and non-users.
2. **Promote Inclusion:** Reduce social exclusion for deaf individuals.
3. **Improve Accessibility:** Make communication more accessible in various settings.
4. **User-Friendly:** Ensure the application is easy to use across multiple devices.

## Solution Overview

The web application captures real-time video and translates Arabic sign language gestures into text using a machine learning model. The solution involves several key components:

### Technical Components

1. **Machine Learning Model:**
   - **Architecture:** CNN-LSTM model.
   - **Framework:** TensorFlow and Keras.
   - **Functionality:** Recognizes Arabic sign language gestures by analyzing video frames and extracting keypoints.

2. **Dataset:**
   - **Name:** KArSL dataset.
   - **Content:** 502 isolated sign words collected using Microsoft Kinect V2, performed by three professional signers.

3. **Model Architecture:**
   - **Bidirectional LSTM:** Captures temporal dependencies in sign language sequences.
   - **Dense Layers:** Classify the extracted features into 89 different sign language classes.
   - **Output Layer:** Uses softmax activation for classification.

4. **Web Application:**
   - **Backend:** Django (Python).
   - **Frontend:** HTML, CSS, JavaScript.
   - **Functionality:** Captures video, processes it through the machine learning model, and displays the translated text in real-time.

### Results and Performance

- **Accuracy:** Achieved high training and test accuracies, demonstrating the model's effectiveness.
- **User Interface:** Developed a user-friendly web interface to interact with the model and provide real-time translations.

### Challenges and Solutions

- **Uneven Frame Counts:** Managed uneven frame counts in video data for consistent processing.
- **Keypoint Optimization:** Optimized the number of keypoints for efficient gesture recognition.

## How to Use

1. **Setup:**
   - Clone the repository from GitHub.
   - Install required dependencies using `pip install -r requirements.txt`.

2. **Run the Application:**
   - Start the Django server using `python manage.py runserver`.
   - Access the web application through the provided local server address.

3. **Using the Application:**
   - Allow camera access for the web application.
   - Perform sign language gestures in front of the camera.
   - The application will display the translated text in real-time.

## Future Work

- **Expand Dataset:** Include more sign language gestures and words to enhance the model’s vocabulary.
- **Improve Accuracy:** Refine the model to achieve even higher accuracy and reliability.
- **Mobile Application:** Develop a mobile version of the application for greater accessibility.

## Conclusion

This project demonstrates the potential of machine learning and computer vision to create impactful solutions for real-world problems. By translating Arabic sign language into text, we aim to improve communication and inclusion for deaf individuals, making a positive difference in their lives.


I appreciate your interest in our project. We look forward to your feedback and support!

---

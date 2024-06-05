import mediapipe as mp
import numpy as np
from PIL import Image
import os
import warnings
import cv2
warnings.filterwarnings("ignore")

N_FRAMES = 60 # N Frames Per Prediction
N_KEYPOINTS = 225 # N Keypoints captured by mediapipe

# Maps local labels to Arabic Translations
arabic_labels = {
    0: 'يدعم',  # supports
    1: 'يدخن',  # smokes
    2: 'يكسر',  # breaks
    3: 'يحب',   # loves
    4: 'يشوي',  # grills
    5: 'يحصد',  # harvests
    6: 'يحرث',  # plows
    7: 'يكره',  # hates
    8: 'يستحم', # bathes
    9: 'يسقي',  # irrigates
    10: 'يساعد', # helps
    11: 'يبني',  # builds
    12: 'يمشي',  # walks
    13: 'يتنامى', # grows
    14: 'يختار', # chooses
    15: 'يفكر',  # thinks
    16: 'ينادي', # calls
    17: 'يصبغ',  # dyes
    18: 'يقف',   # stands
    19: 'ث',
    20: '100',
    21: 'غ',
    22: 'ظ',
    23: 'ر',
    24: '500',
    25: '0',
    26: '6',
    27: 'ال',
    28: '1000000',
    29: 'ى',
    30: 'ح',
    31: 'د',
    32: '10000000',
    33: 'ش',
    34: 'س',
    35: 'إ',
    36: '70',
    37: '10',
    38: '90',
    39: '4',
    40: 'ؤ',
    41: 'ق',
    42: 'ي',
    43: '40',
    44: '60',
    45: '8',
    46: 'خ',
    47: 'ص',
    48: 'آ',
    49: 'ب',
    50: '200',
    51: 'ج',
    52: '400',
    53: 'م',
    54: 'ه',
    55: 'ئـ',
    56: '9',
    57: '300',
    58: '700',
    59: '3',
    60: 'ط',
    61: '80',
    62: 'ك',
    63: '50',
    64: 'ء',
    65: 'ز',
    66: 'ذ',
    67: 'أ',
    68: '30',
    69: 'لا',
    70: 'ت',
    71: '1000',
    72: '7',
    73: 'ل',
    74: 'ئ',
    75: 'أ',
    76: 'ف',
    77: 'و',
    78: '600',
    79: 'ا',
    80: '2',
    81: 'ع',
    82: '20',
    83: '1',
    84: 'ة',
    85: 'لا',
    86: 'ض',
    87: 'ن',
    88: '800'
}

# Maps folder numbers (KArSL Indices to local labels)
actions = {
    '0184': 0,
    '0183': 1,
    '0172': 2,
    '0174': 3,
    '0176': 4,
    '0179': 5,
    '0177': 6,
    '0175': 7,
    '0190': 8,
    '0180': 9,
    '0182': 10,
    '0171': 11,
    '0173': 12,
    '0187': 13,
    '0185': 14,
    '0181': 15,
    '0186': 16,
    '0188': 17,
    '0189': 18,
    '0035': 19,
    '0020': 20,
    '0050': 21,
    '0048': 22,
    '0041': 23,
    '0024': 24,
    '0001': 25,
    '0007': 26,
    '0070': 27,
    '0030': 28,
    '0068': 29,
    '0037': 30,
    '0039': 31,
    '0031': 32,
    '0044': 33,
    '0043': 34,
    '0066': 35,
    '0011': 37,
    '0019': 38,
    '0005': 39,
    '0062': 40,
    '0052': 41,
    '0059': 42,
    '0014': 43,
    '0016': 44,
    '0009': 45,
    '0038': 46,
    '0045': 47,
    '0067': 48,
    '0033': 49,
    '0021': 50,
    '0036': 51,
    '0023': 52,
    '0055': 53,
    '0057': 54,
    '0064': 55,
    '0010': 56,
    '0022': 57,
    '0026': 58,
    '0004': 59,
    '0047': 60,
    '0018': 61,
    '0053': 62,
    '0015': 63,
    '0065': 64,
    '0042': 65,
    '0040': 66,
    '0013': 68,
    '0017': 69,
    '0034': 70,
    '0029': 71,
    '0008': 72,
    '0054': 73,
    '0063': 74,
    '0061': 75,
    '0051': 76,
    '0058': 77,
    '0025': 78,
    '0032': 79,
    '0003': 80,
    '0049': 81,
    '0012': 82,
    '0002': 83,
    '0060': 84,
    '0069': 85,
    '0046': 86,
    '0056': 87,
    '0027': 88,
}

class DataLoader:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    X, Y = [], []

    # Method for extracting the keypoints in images
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        nose = pose[:3]
        lh_wrist = lh[:3]
        rh_wrist = rh[:3]
        
        pose_adjusted = DataLoader.adjust_landmarks(pose, nose)
        lh_adjusted = DataLoader.adjust_landmarks(lh, lh_wrist)
        rh_adjusted = DataLoader.adjust_landmarks(rh, rh_wrist)
        
        return np.concatenate((pose_adjusted, lh_adjusted, rh_adjusted))


    # Drawing the key points on images to see the results
    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 ) 

    def adjust_landmarks(arr, center):
        # Reshape the array to have shape (n, 3)
        arr_reshaped = arr.reshape(-1, 3)

        # Repeat the center array to have shape (n, 3)
        center_repeated = np.tile(center, (len(arr_reshaped), 1))

        # Subtract the center array from the arr array
        arr_adjusted = arr_reshaped - center_repeated

        # Reshape arr_adjusted back to shape (n*3,)
        arr_adjusted = arr_adjusted.reshape(-1)
        
        return arr_adjusted
    
    def load_data(frames_dir):
        # List to store results
        results_list = []

        # Loop through frames in the directory
        for filename in sorted(os.listdir(frames_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
                # Read the image
                image_path = os.path.join(frames_dir, filename)
                image = np.array(Image.open(image_path).convert('RGB'))

                # Convert image to RGB format (if it's not already in RGB)
                if image.shape[-1] == 1:  # Grayscale image
                    image = np.repeat(image, 3, axis=-1)  # Convert to RGB by repeating the grayscale channel

                # Detect landmarks using MediaPipe Holistic
                with DataLoader.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    results = holistic.process(image)
                    results_extracted = DataLoader.extract_keypoints(results)

                # Append results to the results list
                results_list.append(results_extracted)
        
        if len(results_list) > N_FRAMES: return

        # Convert results list to NumPy array
        while len(results_list) < N_FRAMES:
            results_list.append(np.zeros((N_KEYPOINTS)))

        results_array = np.array(results_list)

        DataLoader.X.append(results_array)
        DataLoader.Y.append(frames_dir)
    
    def load_inference_data(path):
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {path}")
            return
        
        results_list = []
        while True:
            ret, image = cap.read()

            if not ret:
                print("End of video file reached or no frames to read")
                break
            
            # Convert image to RGB format (if it's not already in RGB) using opencv (cv2)   
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Convert image to RGB format (if it's not already in RGB)
            if image.shape[-1] == 1:  # Grayscale image
                image = np.repeat(image, 3, axis=-1)  # Convert to RGB by repeating the grayscale channel

            # Detect landmarks using MediaPipe Holistic
            with DataLoader.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                results = holistic.process(image)
                results_extracted = DataLoader.extract_keypoints(results)

            # Append results to the results list
            results_list.append(results_extracted)

        if len(results_list) > N_FRAMES: results_list = results_list[:N_FRAMES]

        while len(results_list) < N_FRAMES:
            results_list.append(np.zeros((N_KEYPOINTS)))
        
        results_array = np.array(results_list)
        results_array = np.expand_dims(results_array, axis=0)

        return results_array if results_array is not None else np.zeros((1, N_FRAMES, N_KEYPOINTS))
    
    
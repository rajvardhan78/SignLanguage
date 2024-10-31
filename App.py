import logging

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from deep_translator import GoogleTranslator
from tensorflow.keras.models import load_model
import pickle
import google.generativeai as genai

class SignLanguageDetector:
    def __init__(self, model_path='best_model.h5', label_map_path='cnn_dataset.pickle'):
        """Initialize the detector with model and MediaPipe hands"""
        # Load model
        self.model = load_model(model_path)

        # Load label map
        with open(label_map_path, 'rb') as f:
            dataset = pickle.load(f)
            self.label_map = {v: k for k, v in dataset['label_map'].items()}

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Constants
        self.IMG_SIZE = (128, 128)
        self.NUM_LANDMARKS = 21
        self.BUFFER_SIZE = 5  # Number of consecutive same predictions needed
        self.NO_HANDS_THRESHOLD = 15  # Number of frames without hands needed to trigger Gemini

        # Initialize state for tracking frames without hands
        self.no_hands_counter = 0
        self.label_buffer = []
        self.current_label = None

    def process_frame(self, frame):
        """Process a single frame and return predictions with state management"""
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                self.no_hands_counter += 1
                self.label_buffer = []  # Clear buffer when no hands
                return frame, None

            self.no_hands_counter = 0  # Reset counter when hands detected

            # Extract features and create visualization
            landmark_features = self.extract_landmark_features(results)
            landmark_image = self.create_landmark_image(results, frame.shape)

            # Make prediction
            prediction = self.model.predict(
                [np.expand_dims(landmark_features, 0),
                 np.expand_dims(landmark_image, 0)],
                verbose=0
            )
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            # Update label buffer
            predicted_label = self.label_map[predicted_class]
            self.label_buffer.append(predicted_label)

            # Only keep last BUFFER_SIZE predictions
            if len(self.label_buffer) > self.BUFFER_SIZE:
                self.label_buffer.pop(0)

            # Check if we have consistent predictions
            if len(self.label_buffer) == self.BUFFER_SIZE and \
                    all(x == self.label_buffer[0] for x in self.label_buffer) and \
                    self.label_buffer[0] != self.current_label:
                self.current_label = self.label_buffer[0]
                return frame, (self.current_label, confidence, True)  # True indicates new label

            return frame, (self.current_label, confidence, False) if self.current_label else None

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame, None

    def extract_landmark_features(self, results):
        """Extract normalized landmark features from MediaPipe results"""
        features = np.zeros((2, self.NUM_LANDMARKS, 3))

        if results.multi_hand_landmarks:
            hands_data = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_mean = np.mean([lm.x for lm in hand_landmarks.landmark])
                hands_data.append((x_mean, hand_landmarks))
            hands_data.sort(key=lambda x: x[0])

            for hand_idx, (_, hand_landmarks) in enumerate(hands_data):
                if hand_idx >= 2:
                    break
                for i, landmark in enumerate(hand_landmarks.landmark):
                    features[hand_idx, i] = [landmark.x, landmark.y, landmark.z]

        return features

    def create_landmark_image(self, results, img_shape):
        """Create a visualization of hand landmarks"""
        canvas = np.zeros((*self.IMG_SIZE, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=2)
                )

        return canvas



def get_gemini_response(detected_labels):
    """Get response from Gemini AI"""
    with st.spinner('Generating sentence from signs...'):
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')

        prompt = f"""
        Create a natural, grammatically correct and contextually relevant sentence using these sign language words: {', '.join(detected_labels)}.
        The sentence should be simple and make sense in a casual introduction and don't add other extra words in the sentence unless it's necessary.
        Only return the sentence, nothing else. For nouns, use the passed letters as it is unless it is very near to a known noun.
        """

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating sentence: {str(e)}"


def translate_text(text, target_language):
    """Translate text to the target language using deep-translator."""
    with st.spinner('Translating sentence...'):
        try:
            translation = GoogleTranslator(target=target_language).translate(text)
            return translation
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return "Translation failed."


def show_instructions():
    """Display the instructions for using the application"""
    st.markdown("""
    ### Instructions:
    1. Click the 'Start Detection' button below
    2. Show hand signs to the camera
    3. The detected sign will be shown and collected
    4. When you pause (no hands detected), The System will generate a sentence using the collected signs
    """)


def main():
    st.set_page_config(page_title="Indian Sign Language Detection", layout="wide")
    st.title("Real-time Indian Sign Language Detection")

    # Initialize session state variables
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    if 'collected_signs' not in st.session_state:
        st.session_state.collected_signs = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    def load_detector():
        return SignLanguageDetector()

    def stop_detection():
        st.session_state.detection_running = False
        st.session_state.collected_signs = []
        # Force rerun to update the UI
        st.rerun()

    try:
        with st.spinner('Loading sign language detection model...'):
            detector = load_detector()

        # Only show instructions when detection is not running
        if not st.session_state.detection_running:
            show_instructions()

        col1, col2 = st.columns([2, 1])

        with col1:
            video_placeholder = st.empty()

        with col2:
            st.markdown("### Detected Sign:")
            prediction_container = st.container()
            with prediction_container:
                prediction_placeholder = st.empty()
                confidence_bar = st.empty()

            st.markdown("### Collected Signs:")
            collected_signs_placeholder = st.empty()

            st.markdown("### Generated Sentence:")
            sentence_container = st.container()
            with sentence_container:
                sentence_placeholder = st.empty()

            st.markdown("### Translated Sentence:")
            translation_container = st.container()
            with translation_container:
                translated_sentence_placeholder = st.empty()

            # Add a dropdown for selecting Indian languages
            languages = {
                'Hindi': 'hi',
                'Bengali': 'bn',
                'Telugu': 'te',
                'Marathi': 'mr',
                'Tamil': 'ta',
                'Gujarati': 'gu',
                'Kannada': 'kn',
                'Malayalam': 'ml',
                'Punjabi': 'pa',
                'Odia': 'or'
            }
            selected_language = st.selectbox("Select Language", list(languages.keys()))

        # Create toggle button for detection
        if not st.session_state.detection_running:
            if st.button("Start Detection"):
                st.session_state.detection_running = True
                st.rerun()
        else:
            if st.button("Stop Detection", type="primary", key="stop_button"):
                stop_detection()

        # Main detection loop
        if st.session_state.detection_running:
            cap = cv2.VideoCapture(0)
            try:
                while st.session_state.detection_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break

                    # Process frame with spinner
                    with prediction_container:
                        with st.spinner('Processing...'):
                            processed_frame, prediction = detector.process_frame(frame)
                            video_placeholder.image(
                                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_column_width=True
                            )

                    if prediction:
                        sign, confidence, is_new_label = prediction
                        prediction_placeholder.markdown(f"# {sign}")
                        confidence_bar.progress(float(confidence),
                                                text=f"Confidence: {confidence:.2%}")

                        if is_new_label and sign not in st.session_state.collected_signs:
                            st.session_state.collected_signs.append(sign)
                    else:
                        prediction_placeholder.markdown("# No hands detected")
                        confidence_bar.empty()

                        if detector.no_hands_counter >= detector.NO_HANDS_THRESHOLD and st.session_state.collected_signs:
                            # Generate sentence with spinner
                            with sentence_container:
                                with st.spinner('Generating sentence...'):
                                    sentence = get_gemini_response(st.session_state.collected_signs)
                                    sentence_placeholder.markdown(f"#### {sentence}")

                            # Translate sentence with spinner
                            with translation_container:
                                with st.spinner('Translating sentence...'):
                                    if sentence:
                                        translated_sentence = translate_text(sentence, languages[selected_language])
                                    else:
                                        translated_sentence = "No sentence to translate."
                                    translated_sentence_placeholder.markdown(f"#### {translated_sentence}")

                            st.session_state.collected_signs = []  # Clear collected signs
                            detector.no_hands_counter = 0  # Reset counter

                    collected_signs_placeholder.markdown(
                        ", ".join(st.session_state.collected_signs) if st.session_state.collected_signs
                        else "No signs collected yet"
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                cap.release()

    except Exception as e:
        st.error(f"Failed to initialize detector: {str(e)}")


if __name__ == "__main__":
    main()
import os  # Used to interact with the operating system (e.g., check file existence).
import wave  # Used to handle audio file operations in WAV format.
import time  # Provides time-related functions (e.g., adding delays).
import sounddevice as sd  # For capturing and streaming audio input/output.
from vosk import Model, KaldiRecognizer  # Vosk library for offline speech recognition.
import pvporcupine  # Porcupine library for wake word detection.
import numpy as np  # Used for numerical and audio data processing.
import json  # Used for parsing JSON data from speech recognition results.

"""
    To make this program work, you need to get the Vosk model and
    PVPorcupine model. 
    
    https://alphacephei.com/vosk/models
    https://picovoice.ai/platform/porcupine/
    To get free porcupine wake word detection, you sign up at the
    link above and download the WINDOWS platform.
    
    Make sure the models are referenced in the relevant variables below.
    
    You can make a file inside of the project or outside, but if outside 
    of the project make sure it is referenced with the absolute path
"""

# Path to Vosk model
VOSK_MODEL = "path/to/vosk/model"  # Directory for the Vosk speech recognition model.

# Path to Porcupine model
PORCUPINE_MODEL = "path/to/porcupine/ppn"  # File path for Porcupine's wake word model.

# Check if the Vosk model exists
if not os.path.exists(VOSK_MODEL):
    raise Exception("Vosk model not found! Please download it and place it in './models'.")
    # Ensures the program doesn't proceed if the Vosk model is missing.

# Check if the Porcupine model exists
if not os.path.exists(PORCUPINE_MODEL):
    raise Exception("Porcupine model not found! Please download it and place it in './models'.")
    # Ensures the program doesn't proceed if the Porcupine model is missing.

# Initialize the Vosk speech recognition model
model = Model(VOSK_MODEL)  # Loads the Vosk model from the specified directory.
recognizer = KaldiRecognizer(model, 16000)  # Initializes the recognizer for 16kHz audio input.

# Porcupine setup for wake word detection
porcupine = pvporcupine.create(
    access_key="access key",  # Porcupine API access key.
    keyword_paths=[PORCUPINE_MODEL],  # Links to the Porcupine wake word model file.
    sensitivities=[0.5]  # Sets the sensitivity for detecting the wake word.
)
CHUNK_SIZE = porcupine.frame_length  # Specifies the audio frame size Porcupine requires.

# Audio configuration constants
SAMPLE_RATE = 16000  # Sample rate for recording audio, matching Vosk and Porcupine's requirements.
SILENCE_THRESHOLD = 1.0  # Threshold duration (in seconds) to detect silence after speech.


def listen_for_wake_word():
    """
    Listens for a wake word using Porcupine. Returns True if detected.
    """
    # Open an audio input stream for continuous listening
    with sd.InputStream(
            samplerate=SAMPLE_RATE,  # Matches the configured sample rate.
            channels=1,  # Ensures mono audio input.
            dtype="int16",  # 16-bit PCM audio format.
            blocksize=porcupine.frame_length  # Block size matches Porcupine's frame length.
    ) as stream:
        while True:
            try:
                # Read audio data matching the frame length
                pcm_data, overflow = stream.read(porcupine.frame_length)
                if overflow:
                    print("Audio buffer overflow detected!")  # Alerts if audio input overflows.

                # Convert audio data to a numpy array
                pcm_array = np.frombuffer(pcm_data, dtype="int16")

                # Check frame length consistency
                if len(pcm_array) != porcupine.frame_length:
                    print(f"Incorrect frame length! Expected {porcupine.frame_length}, got {len(pcm_array)}")
                    continue

                # Process the audio frame for wake word detection
                keyword_index = porcupine.process(pcm_array)
                if keyword_index >= 0:  # If the wake word is detected
                    print("Wake word detected!")  # Notify wake word detection.
                    return True
            except Exception as e:
                print(f"Error in wake word detection: {e}")  # Catch and log errors.


def recognize_command_terms(audioOutput):
    """
    Matches recognized audio output to predefined commands and returns a response.
    """
    match audioOutput:  # Simplifies command matching using Python's structural pattern matching.
        case "turn on the lights":
            return "Lights have been turned on"
        case "turn off the lights":
            return "Lights have been turned off"
        case "activate the house alarm":
            return "House alarm is activated"
        case "lock the door":
            return "The doors are locked"
        case "roll down the blinds":
            return "Blinds are rolled down"
        case "roll up the blinds":
            return "Blinds are rolled up"
        case _:
            return "Command not recognized"  # Default response for unrecognized commands.


def record_and_recognize():
    """
    Records audio input and recognizes speech using Vosk.
    """
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)  # Reinitialize recognizer for fresh recognition.
        audio_data = []  # Stores recorded audio frames.

        while True:
            pcm_data, overflow = stream.read(4000)  # Read audio in chunks.
            if overflow:
                print("Audio buffer overflow detected!")  # Warn about buffer overflow.
            pcm_array = np.frombuffer(pcm_data, dtype="int16")  # Convert chunk to numpy array.
            audio_data.append(pcm_array)  # Append the chunk to audio data.
            data_bytes = pcm_array.tobytes()  # Convert numpy array to bytes for Vosk.

            if recognizer.AcceptWaveform(data_bytes):  # Check if Vosk detects end of a sentence.
                print("End of sentence detected.")
                break

        final_result = recognizer.Result()  # Get final recognition result from Vosk.
        print(f"Recognition result: {final_result}")

        # Parse the result from JSON
        result_dict = json.loads(final_result)  # Converts JSON string to Python dictionary.
        recognized_text = result_dict.get("text", "")  # Extract recognized text safely.

        # Process recognized text with command handler
        response = recognize_command_terms(recognized_text)
        print(f"Command response: {response}")
        return response


def save_audio(audio_data, sample_rate):
    """
    Saves recorded audio data to a WAV file.
    """
    file_name = "output.wav"  # Name of the output file.
    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(1)  # Configures mono audio.
        wf.setsampwidth(2)  # Sets 16-bit audio width.
        wf.setframerate(sample_rate)  # Sets the audio sample rate.
        wf.writeframes(b"".join(audio_data))  # Writes audio frames to the file.


def main():
    """
    Main function to handle wake word detection and sentence recognition.
    """
    print("Program started. Waiting for the wake word...")  # Notify program start.
    while True:
        if listen_for_wake_word():  # Wait for wake word detection.
            print("Wake word detected. Starting sentence recognition...")
            sentence = record_and_recognize()  # Start recording and recognizing speech.
            print("Response:", sentence)  # Print the response from recognized command.


if __name__ == "__main__":
    try:
        main()  # Run the main function.
    except KeyboardInterrupt:
        print("Exiting program.")  # Gracefully handle program interruption.
    finally:
        if porcupine:
            porcupine.delete()  # Clean up Porcupine resources.

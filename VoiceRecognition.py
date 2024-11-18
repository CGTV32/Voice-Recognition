import os
import wave
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pvporcupine
import numpy as np

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
VOSK_MODEL = "name/of/vosk/folder"
# Path to Porcupine model
PORCUPINE_MODEL = "name/of/porcupine/ppn"

# Check if the model exists
if not os.path.exists(VOSK_MODEL):
    raise Exception("Vosk model not found! Please download it and place it in './models'.")

if not os.path.exists(PORCUPINE_MODEL):
    raise Exception("Porcupine model not found! Please download it and place it in './models'.")

# Initialize the Vosk model and recognizer
model = Model(VOSK_MODEL)
recognizer = KaldiRecognizer(model, 16000)

# Porcupine setup
porcupine = pvporcupine.create(
    access_key="Your access key for porcupine",
    keyword_paths = [PORCUPINE_MODEL],
    sensitivities=[0.5]
)
CHUNK_SIZE = porcupine.frame_length

# Audio parameters
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 2.0  # Duration (seconds) to detect silence after speech begins


def listen_for_wake_word():
    """
    Listens for a wake word using Porcupine. Returns True if detected.
    """
    print("Listening for the wake word...")

    # Open an audio stream with the correct configuration
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,  # Ensure mono audio
        dtype="int16",  # 16-bit PCM data
        blocksize=porcupine.frame_length  # Block size matches Porcupine's frame length
    ) as stream:
        while True:
            try:
                # Read exactly porcupine.frame_length samples
                pcm_data, overflow = stream.read(porcupine.frame_length)

                # Flatten the data into a 1D numpy array
                pcm_array = np.frombuffer(pcm_data, dtype="int16")

                # Ensure the data length matches expected frame length
                if len(pcm_array) != porcupine.frame_length:
                    print(f"Warning: Incorrect frame length! Expected {porcupine.frame_length}, got {len(pcm_array)}")
                    continue

                # Pass the raw numpy array directly to Porcupine
                keyword_index = porcupine.process(pcm_array)

                # Check if the wake word was detected
                if keyword_index >= 0:
                    print("Wake word detected!")
                    return True
            except Exception as e:
                print(f"Error in wake word detection: {e}")



def record_and_recognize():
    """
    Records audio input and recognizes sentences using Vosk.
    Returns a recognized sentence or an empty string if no speech is detected.
    """
    print("Recording audio...")

    # Start the audio stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        audio_data = []  # Buffer to store recorded audio

        while True:
            # Read audio data in chunks
            pcm_data, overflow = stream.read(4000)  # Adjust block size if necessary
            if overflow:
                print("Audio buffer overflow detected!")

            # Convert pcm_data to numpy array
            pcm_array = np.frombuffer(pcm_data, dtype="int16")

            # Append raw PCM data to buffer
            audio_data.append(pcm_array)

            # Convert numpy array to bytes for Vosk
            data_bytes = pcm_array.tobytes()

            # Process data using the recognizer
            if recognizer.AcceptWaveform(data_bytes):
                print("End of sentence detected.")
                break

        # Recognize the full sentence
        final_result = recognizer.Result()
        print(f"Recognition result: {final_result}")
        return final_result

    # Save the recorded audio to a WAV file
    save_audio(audio_data, SAMPLE_RATE)

    # Return the final recognition result
    return recognizer.FinalResult()


def save_audio(audio_data, sample_rate):

    #Save the recorded audio to a WAV file.

    file_name = "output.wav"
    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(audio_data))
    print(f"Audio saved to {file_name}")


def main():
    """
    Main function to handle wake word detection and sentence recognition.
    """
    print("Program started. Waiting for the wake word...")
    while True:
        if listen_for_wake_word():
            print("Wake word detected. Starting sentence recognition...")
            sentence = record_and_recognize()
            print("Final result:", sentence)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        if porcupine:
            porcupine.delete()

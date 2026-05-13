import wave
import numpy as np

def convert_to_mono(input_filepath, output_filepath):
    # 1. Read the original WAV
    with wave.open(input_filepath, 'rb') as wav_in:
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        framerate = wav_in.getframerate()
        frames = wav_in.readframes(wav_in.getnframes())

    if n_channels == 1:
        print("File is already Mono! The issue might be PyGame running in the background.")
        return

    print(f"Original file has {n_channels} channels. Converting...")

    # 2. Convert raw bytes to a numpy array
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 1:
        dtype = np.uint8
    else:
        print("Unsupported bit depth. Please use an online converter.")
        return

    audio_data = np.frombuffer(frames, dtype=dtype)
    
    # 3. Reshape array and average the left and right channels together
    audio_data = audio_data.reshape(-1, n_channels)
    mono_data = audio_data.mean(axis=1).astype(dtype)

    # 4. Save the new Mono WAV file
    with wave.open(output_filepath, 'wb') as wav_out:
        wav_out.setnchannels(1) # Force 1 channel
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(framerate)
        wav_out.writeframes(mono_data.tobytes())
        
    print(f"Success! Saved mono file to {output_filepath}")

if __name__ == "__main__":
    # Change these paths to point to your actual audio file
    original_file = "assets/audio/engine.wav"
    fixed_file = "assets/audio/engine_mono.wav"
    
    convert_to_mono(original_file, fixed_file)
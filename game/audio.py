import numpy as np
import sounddevice as sd

class AudioManager:
    def __init__(self, sample_rate=44100, warp_factor=2.0):
        self.fs = sample_rate
        self.stream = sd.OutputStream(samplerate=self.fs, channels=2, callback=self._callback)
        
        # State variables updated by the Ego class
        self.speed = 0.0
        self.ratio = 0.5
        self.is_running = False
        
        # Internal phase tracking to prevent 'clicks' between buffers
        self.phase = 0.0

        # Store the parameter in the class state
        # 1.0 = Linear (No safe zone)
        # 2.0 = Squared (Standard safe zone)
        # 3.0+ = Extreme (Massive safe zone, violent edge warnings)
        self.warp_factor = warp_factor

    def _callback(self, outdata, frames, time, status):
        """This is the high-priority thread that generates audio buffers."""
        # Calculate fundamental frequency based on speed (e.g., 50Hz to 400Hz)
        freq = 50 + (self.speed * 5.0) 
        
        # Create time array for this buffer
        t = (np.arange(frames) / self.fs)
        
        # Generate a Sawtooth wave (classic engine sound)
        # We track phase manually to keep the wave continuous across callbacks
        cycles = freq * t + self.phase
        wave = 2 * (cycles - np.floor(0.5 + cycles))
        self.phase = cycles[-1] % 1.0
        
        # Apply Equal Power Panning
        rad_clamped = max(0.0, min(1.0, self.ratio))

        # Warp the pan value to make it more perceptually linear
        pan_x = (rad_clamped - 0.5) * 2.0
        pan_warped = np.sign(pan_x) * (np.abs(pan_x) ** self.warp_factor)
        rad_warped = (pan_warped + 1.0) / 2.0

        # Calculate Equal Power gains using the new warped ratio
        left_gain = np.cos((np.pi / 2.0) * rad_warped)
        right_gain = np.sin((np.pi / 2.0) * rad_warped)
        
        # Master volume (throttle)
        volume = 0.3
        
        outdata[:, 0] = wave * left_gain * volume  # Left Channel
        outdata[:, 1] = wave * right_gain * volume # Right Channel

    def start(self):
        self.stream.start()
        self.is_running = True

    def update_state(self, speed, ratio):
        """Called by your main loop to feed simulation data to the audio thread."""
        self.speed = speed
        self.ratio = ratio

    def stop(self):
        self.stream.stop()
        self.is_running = False

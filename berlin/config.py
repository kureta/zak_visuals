from dataclasses import dataclass


@dataclass
class Config:
    fps = 25
    sample_rate = 44100
    frame_size = 2048
    frames_start_index = 378
    frames_end_index = 24228
    audio_dir = 'data/audio'
    audio_extensions = ['.aac']
    image_dir = 'data/images'
    image_extensions = ['.png']

    @property
    def hop_size(self):
        return self.sample_rate // self.fps

    @property
    def spectrum_size(self):
        return self.frame_size // 2 + 1

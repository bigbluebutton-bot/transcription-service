# m_convert_audio.py
import subprocess
import numpy as np

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import data
import logger

log = logger.get_logger()

class Convert_Audio(Module):
    def __init__(self,
                    convert_sample_rate: int = 16000
                ) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Convert-Audio-Module"
        )
        self.convert_sample_rate: int = convert_sample_rate

    def init_module(self) -> None:
        pass

    def load_audio_from_binary(self, data: bytes) -> np.ndarray:
        """
        Process binary audio data (e.g., OGG Opus) and convert it to a mono waveform, resampling as necessary.

        Parameters
        ----------
        data: bytes
            The binary audio data to process.

        sr: int
            The sample rate to resample the audio if necessary.

        Returns
        -------
        np.ndarray
            A NumPy array containing the audio waveform, in float32 dtype.
        """
        try:
            # Set up the ffmpeg command to read from a pipe
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i", "pipe:0",  # Use the pipe:0 to read from stdin
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(self.convert_sample_rate),
                "-",
            ]

            # Run the ffmpeg process, feeding it the binary data through stdin
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = process.communicate(input=data)

            if process.returncode != 0:
                raise RuntimeError(f"Failed to load audio: {err.decode()}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        # Convert the raw audio data to a NumPy array and normalize it
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def execute(
        self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule
    ) -> None:
        if not dp.data:
            raise Exception("No data found")
        if not dp.data.raw_audio_data:
            raise Exception("No audio data found")

        audio_data = self.load_audio_from_binary(dp.data.raw_audio_data)
        dp.data.audio_data_sample_rate = self.convert_sample_rate
        dp.data.audio_data = audio_data

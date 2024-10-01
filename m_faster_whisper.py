# m_faster_whisper.py
from typing import List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline  # type: ignore
# from whisperx.audio import log_mel_spectrogram  # type: ignore
import torch

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions
import logger
import data


log = logger.get_logger()

class Faster_Whisper_transcribe(Module):
    def __init__(self,
                    model_path: str = ".models/faster-whisper",
                    model_size: str = "large-v3",
                    task: str = "transcribe",
                    compute_type: str = "float16", # "float16" or "int8"
                    batching: bool = True,
                    batch_size: int = 32,
                    device: str = "cuda"
                ) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="Whisper-Module"
        )
        self.model_path: str = model_path
        self.model_size: str = model_size
        self.task: str = task
        self.compute_type: str = compute_type # "float16" or "int8"
        self.batching: bool = batching
        self.batch_size: int = batch_size
        self.device: str = device if torch.cuda.is_available() else "cpu"
        
        self._model: Optional[WhisperModel] = None
        self._batched_model: Optional[BatchedInferencePipeline] = None

    def init_module(self) -> None:
        log.info(f"Loading model whisper:'{self.model_size}'...")
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type, download_root=self.model_path)
        if self.batching:
            self._batched_model = BatchedInferencePipeline(model=self._model)
        log.info("Whisper model loaded")

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self._model:
            raise Exception("Whisper model not loaded")
        if not dp.data:
            raise Exception("No data found")
        if dp.data.audio_data is None:
            raise Exception("No audio data found")
        if not dp.data.vad_result and self.batching:
            raise Exception("No audio data from VAD found")
        if dp.data.audio_buffer_start_after is None:
            raise Exception("No audio buffer start after found")
        
        
        audio_buffer_start_after = dp.data.audio_buffer_start_after
        audio = dp.data.audio_data
        if self.batching and self._batched_model is not None:
            segments, info = self._batched_model.transcribe(
                                        audio,
                                        batch_size=self.batch_size,
                                        task=self.task,
                                        word_timestamps=True,
                                        vad_segments=dp.data.vad_result
                                    )
        else:
            segments, info = self._model.transcribe(
                                    audio,
                                    task=self.task,
                                    word_timestamps=True
                                )

        result = []
        for segment in segments:
            words = []
            # print(f"Segment: {segment}")
            if segment.words:
                for word in segment.words:
                    w = data.Word(
                        word=word.word,
                        start=word.start + audio_buffer_start_after,
                        end=word.end + audio_buffer_start_after,
                        probability=word.probability
                    )
                    words.append(w)

            ts = data.TextSegment(
                text=segment.text,
                start=segment.start + audio_buffer_start_after,
                end=segment.end + audio_buffer_start_after,
                words=words
            )
            result.append(ts)
        dp.data.transcribed_segments = result
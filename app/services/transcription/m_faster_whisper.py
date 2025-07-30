# m_faster_whisper.py
from __future__ import annotations
import threading
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Any

import numpy as np
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline  # type: ignore

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions
import logger
import data

log = logger.get_logger()
T = TypeVar('T', bound='Faster_Whisper_transcribe')

###############################################################################
# New Singleton AI Manager
###############################################################################
class WhisperAIManager:
    _instance: Optional[WhisperAIManager] = None
    _initialized: bool = False

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Dont deep copy the singleton instance
        return self

    def __new__(cls: Type["WhisperAIManager"], *args: Any, **kwargs: Any) -> "WhisperAIManager":
        if cls._instance is None:
            cls._instance = super(WhisperAIManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: str = ".models/faster-whisper",
        model_size: str = "base",
        compute_type: str = "float16",
        batching: bool = True,
        batch_size: int = 32,
        devices: Optional[List[str]] = None,
    ) -> None:
        if self._initialized:
            return

        self._model_path = model_path
        self._model_size = model_size
        self._compute_type = compute_type
        self._batching = batching
        self._batch_size = batch_size

        if devices is None:
            if torch.cuda.is_available():
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                if len(devices) == 1:
                    devices = ["cuda"]
            else:
                devices = ["cpu"]
        self._devices = devices

        self._devices_mutex: List[threading.Lock] = [threading.Lock() for _ in range(len(self._devices))]
        self._models: Dict[str, WhisperModel] = {}
        self._batched_models: Dict[str, BatchedInferencePipeline] = {}

        self._initialized = True

    def load_model(self) -> None:
        if self._models:
            log.info("Whisper model already loaded")
            return

        log.info(f"Loading Whisper model: {self._model_size}")
        for device in self._devices:
            self._models[device] = WhisperModel(
                self._model_size,
                device=device,
                compute_type=self._compute_type,
                download_root=self._model_path,
            )
            if self._batching:
                self._batched_models[device] = BatchedInferencePipeline(model=self._models[device])
        log.info("Whisper model loaded")

    def unload_model(self) -> None:
        for device in self._devices:
            self._models.pop(device, None)
            self._batched_models.pop(device, None)

    def _get_free_device(self) -> Optional[str]:
        for i, mutex in enumerate(self._devices_mutex):
            if mutex.acquire(blocking=False):
                return self._devices[i]
        return None

    def _release_device(self, device: str) -> None:
        for i, d in enumerate(self._devices):
            if d == device:
                self._devices_mutex[i].release()
                break

    def _execute(self, audio: np.ndarray, task: str, clip_timestamps: Optional[List[Dict[str, int]]] = None) -> Tuple[List[WhisperModel.TranscriptionSegment], Dict]:
        if not self._models:
            self.load_model()

        device: Optional[str] = None
        try:
            while True:
                device = self._get_free_device()
                if device is not None:
                    break
                current_thread = threading.current_thread()
                if hasattr(current_thread, 'timed_out') and current_thread.timed_out:
                    raise TimeoutError("Timeout while waiting for a free device")

            if self._batching and self._batched_models:
                segments, info = self._batched_models[device].transcribe(
                    audio,
                    batch_size=self._batch_size,
                    task=task,
                    word_timestamps=True,
                    vad_filter=True,  # Enable VAD
                    clip_timestamps=clip_timestamps,  # Pass VAD configuration
                )
            else:
                segments, info = self._models[device].transcribe(
                    audio,
                    task=task,
                    word_timestamps=True,
                )

            segments_list = [segment for segment in segments]

            self._release_device(device)
            return segments_list, info
        except Exception as e:
            if device:
                self._release_device(device)
            raise e

    def transcribe(self, audio: np.ndarray, clip_timestamps: Optional[List[Dict[str, int]]] = None) -> Tuple[List[WhisperModel.TranscriptionSegment], Dict]:
        return self._execute(audio, "transcribe", clip_timestamps)

    def translate(self, audio: np.ndarray, clip_timestamps: Optional[List[Dict[str, int]]] = None) -> Tuple[List[WhisperModel.TranscriptionSegment], Dict]:
        return self._execute(audio, "translate", clip_timestamps)


###############################################################################
# Module Class Using the AI Manager Singleton
###############################################################################
class Faster_Whisper_transcribe(Module):
    def __init__(self,
                    task: str = "transcribe"
                ) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5
            ),
            name="Whisper-Module"
        )
        self._ai_manager: WhisperAIManager = WhisperAIManager()

    def init_module(self) -> None:
        self._ai_manager.load_model()

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data:
            raise Exception("No data found")
        if dp.data.audio_data is None:
            raise Exception("No audio data found")
        if not dp.data.vad_result:
            raise Exception("No audio data from VAD found")
        if dp.data.audio_buffer_start_after is None:
            raise Exception("No audio buffer start after found")
        if dp.data.audio_data_sample_rate is None:
            raise Exception("No sampling rate found")


        audio_buffer_start_after = dp.data.audio_buffer_start_after
        audio = dp.data.audio_data
        clip_timestamps =dp.data.vad_result
        audio_data_sample_rate = dp.data.audio_data_sample_rate


        # modify clip_timestamps to match the new format of faster-whisper
        new_clip_timestamps: Optional[List[Dict[str, int]]] = None
        if clip_timestamps is not None:
            # clip_timestamps = [
            #     {
            #         'start': int(15.04971875 * audio_data_sample_rate),
            #         'end': int(16.24784375 * audio_data_sample_rate),
            #     }
            # ]
            new_clip_timestamps = []
            for ct in clip_timestamps:
                start = ct["start"]
                end = ct["end"]
                # if start or end are not of type float skip this timestamp
                if not isinstance(start, float) or not isinstance(end, float):
                    log.warning(f"Skipping clip timestamp: {ct}")
                    continue

                new_clip_timestamps.append(
                    {
                        "start": int(start * audio_data_sample_rate),
                        "end":   int(end   * audio_data_sample_rate)
                    }
                )


        if dp.data.task == data.Task.TRANSCRIBE:
            segments, info = self._ai_manager.transcribe(audio, new_clip_timestamps)
        elif dp.data.task == data.Task.TRANSLATE:
            segments, info = self._ai_manager.translate(audio, new_clip_timestamps)

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

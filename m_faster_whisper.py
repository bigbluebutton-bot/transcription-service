# m_faster_whisper.py
from __future__ import annotations
import copy
import threading
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Any

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline  # type: ignore
# from whisperx.audio import log_mel_spectrogram  # type: ignore
import torch

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions
import logger
import data

log = logger.get_logger()

T = TypeVar('T', bound='Faster_Whisper_transcribe')

class Faster_Whisper_transcribe(Module):
    _instance: Optional[Faster_Whisper_transcribe] = None
    _initialized: bool = False

    def __new__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        if cls._instance is None:
            cls._instance = super(Faster_Whisper_transcribe, cls).__new__(cls)
        return cls._instance  # type: ignore

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Dont deep copy the singleton instance
        return self

    def __init__(
        self,
        model_path: str = ".models/faster-whisper",
        model_size: str = "base",  # tiny, tiny.en, small, small.en, base, base.en, medium, medium.en, large-v1, large-v2, large-v3
        task: str = "transcribe",
        compute_type: str = "float16",  # "float16" or "int8"
        batching: bool = True,
        batch_size: int = 32,
        devices: List[str] = ["all"]
    ) -> None:
        """
        Parameters:
            model_path: Path to the model.
            model_size: Model size (tiny, tiny.en, small, small.en, base, base.en, medium, medium.en, large, etc.)
            task: Task to perform (transcribe, translate).
            compute_type: Compute type ("float16" or "int8").
            batching: Use batching.
            batch_size: Batch size.
            devices: List of devices to use (use ["all"] for all GPUs).
        """
        # Guard to prevent reinitialization on subsequent instantiations.
        if self._initialized:
            return
        
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="Whisper-Module"
        )

        self._model_path = model_path
        self._model_size = model_size
        self._task = task
        self._compute_type = compute_type
        self._batching = batching
        self._batch_size = batch_size

        # Determine devices based on availability.
        if torch.cuda.is_available():
            if devices == ["all"]:
                self._devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                if len(self._devices) == 1:
                    self._devices = ["cuda"]
            else:
                self._devices = devices
        else:
            self._devices = ["cpu"]

        self._devices_mutex: list[threading.Lock] = [threading.Lock() for _ in range(len(self._devices))]

        # Initialize model containers.
        self._models: Dict[str, WhisperModel] = {}
        self._batched_models: Dict[str, BatchedInferencePipeline] = {}

        # Mark the instance as initialized to avoid reinitialization.
        self._initialized = True

    def init_module(self) -> None:
        self.load_model()

    def load_model(self):
        if len(self._models) > 0:
            log.info("Whisper model already loaded")
            return
        log.info("Loading Whisper model")
        for device in self._devices:
            self._models[device] = WhisperModel(
                self._model_size,
                device=device,
                compute_type=self._compute_type,
                download_root=self._model_path
            )
            if self._batching:
                self._batched_models[device] = BatchedInferencePipeline(model=self._models[device])
        log.info("Whisper model loaded")

    def unload_model(self):
        for device in self._devices:
            del self._models[device]
            del self._batched_models[device]

    def _get_free_device(self) -> Optional[str]:
        for i, mutex in enumerate(self._devices_mutex):
            if mutex.acquire(blocking=False):
                return self._devices[i]
        return None
    
    def _release_device(self, device: str) -> None:
        for i, d in enumerate(self._devices):
            if d == device:
                self._devices_mutex[i].release()

    def transcribe(self, audio: np.ndarray, vad_segments: Optional[List[Dict[str, float | List[Tuple[float, float]]]]] = None) -> Tuple[List[WhisperModel.TranscriptionSegment], Dict]:
        # Get free device
        try:
            while True:
                device = self._get_free_device()
                if device:
                    break
                else:
                    # check if thread timed out
                    current_thread = threading.current_thread()
                    if hasattr(current_thread, 'timed_out') and current_thread.timed_out:
                        raise TimeoutError("Timeout while waiting for a free device")
                    
            if self._batching and self._batched_models:
                segments, info = self._batched_models[device].transcribe(
                    audio,
                    batch_size=self._batch_size,
                    task=self._task,
                    word_timestamps=True,
                    vad_segments=vad_segments
                )
            else:
                segments, info = self._models[device].transcribe(
                    audio,
                    task=self._task,
                    word_timestamps=True
                )
            
            # Release device
            self._release_device(device)
            return segments, info
        except Exception as e:
            if device:
                self._release_device(device)
            raise e
        



    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if len(self._models) == 0:
            raise Exception("Whisper model not loaded")
        if not dp.data:
            raise Exception("No data found")
        if dp.data.audio_data is None:
            raise Exception("No audio data found")
        if not dp.data.vad_result and self._batching:
            raise Exception("No audio data from VAD found")
        if dp.data.audio_buffer_start_after is None:
            raise Exception("No audio buffer start after found")
        
        instance_id = dp.pipeline_instance_id
        print(f"Instance in: {instance_id}")
        
        
        audio_buffer_start_after = dp.data.audio_buffer_start_after
        audio = dp.data.audio_data
        segments, info = self.transcribe(audio, dp.data.vad_result)

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
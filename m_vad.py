# m_vad.py
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union
import hashlib
import os
import urllib
from tqdm import tqdm # type: ignore

import numpy as np
import torch
# from whisperx.vad import merge_chunks  # type: ignore
from pyannote.audio import Model # type: ignore
from pyannote.audio.pipelines import VoiceActivityDetection # type: ignore
from pyannote.audio.pipelines.utils import PipelineModel # type: ignore
from pyannote.core import Annotation, Segment, SlidingWindowFeature # type: ignore

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ModuleOptions

import data
import logger

log = logger.get_logger()

# This part was copied and modified from WhisperX's vad.py
# https://github.com/m-bain/whisperX

# Thanks to WhisperX for the amazing work on the VAD model
# Also have a look at the paper:
# @article{bain2022whisperx,
#   title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
#   author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
#   journal={INTERSPEECH 2023},
#   year={2023},
#   comments={Accepted to INTERSPEECH 2023},
#   subjects={Sound (cs.SD); Audio and Speech Processing (eess.AS)},
#   url={https://doi.org/10.48550/arXiv.2303.00747},
#   doi={10.48550/arXiv.2303.00747}
# }

# WhisperX License:
# BSD 2-Clause License

# Copyright (c) 2024, Max Bain

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class VoiceActivitySegmentation(VoiceActivityDetection):
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs: Any,
    ) -> None:
        super().__init__(segmentation=segmentation, fscore=fscore, use_auth_token=use_auth_token, **inference_kwargs)

    def apply(self, audio_waveform: np.ndarray, sr: int = 16000, hook: Optional[Callable] = None) -> SlidingWindowFeature:
        """Apply voice activity detection to a NumPy array waveform.

        Parameters
        ----------
        audio_waveform : np.ndarray
            NumPy array containing the audio waveform.
        sr : int
            Sample rate of the audio waveform.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # If a hook is provided, set it up (e.g., for debugging purposes)
        if hook is not None:
            hook("start", None, file=None)

        # Convert numpy array to PyTorch tensor
        waveform_tensor = torch.tensor(audio_waveform, dtype=torch.float32)

        # Ensure waveform_tensor is 2D: (channel, time)
        if waveform_tensor.ndim == 1:
            # If the audio is mono, add a channel dimension
            waveform_tensor = waveform_tensor.unsqueeze(0)  # Shape becomes (1, time)
        elif waveform_tensor.ndim > 2:
            # If the audio has more than 2 dimensions, it's invalid
            raise ValueError(f"Invalid audio waveform shape: {waveform_tensor.shape}")

        # Prepare the input as a dictionary with waveform (tensor) and sample rate
        input_dict = {
            "waveform": waveform_tensor,
            "sample_rate": sr
        }

        # Process the waveform using the segmentation model
        segmentations: SlidingWindowFeature = self._segmentation(input_dict)

        # Call hook after segmentation step if provided
        if hook is not None:
            hook("segmentation", segmentations, file=None)

        return segmentations





class Binarize:
    """Binarize detection scores using hysteresis thresholding, with min-cut operation
    to ensure not segments are longer than max_duration.

    Parameters
    ----------
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to `onset`.
    min_duration_on : float, optional
        Remove active regions shorter than that many seconds. Defaults to 0s.
    min_duration_off : float, optional
        Fill inactive regions shorter than that many seconds. Defaults to 0s.
    pad_onset : float, optional
        Extend active regions by moving their start time by that many seconds.
        Defaults to 0s.
    pad_offset : float, optional
        Extend active regions by moving their end time by that many seconds.
        Defaults to 0s.
    max_duration: float
        The maximum length of an active segment, divides segment at timestamp with lowest score.
    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.

    Modified by Max Bain to include WhisperX's min-cut operation 
    https://arxiv.org/abs/2303.00747

    Pyannote-audio
    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float('inf')
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores
        Parameters
        ----------
        scores : SlidingWindowFeature
            Detection scores.
        Returns
        -------
        active : Annotation
            Binarized scores.
        """

        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        # annotation meant to store 'active' regions
        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):

            label = k if scores.labels is None else scores.labels[k]

            # initial state
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            t = start
            for t, y in zip(timestamps[1:], k_scores[1:]):
                # currently active
                if is_active: 
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        search_after = len(curr_scores) // 2
                        # divide segment
                        min_score_div_idx = search_after + np.argmin(curr_scores[search_after:])
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx+1:]
                        curr_timestamps = curr_timestamps[min_score_div_idx+1:]
                    # switching from active to inactive
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(y)
                    curr_timestamps.append(t)
                # currently inactive
                else:
                    # switching from inactive to active
                    if y > self.onset:
                        start = t
                        is_active = True

            # if active at the end, add final region
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError("This would break current max_duration param")
            active = active.support(collar=self.min_duration_off)

        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active

class SegmentX:
    def __init__(self, start: float, end: float, speaker: str) -> None:
        self.start: float = start
        self.end: float = end
        self.speaker: str = speaker

from typing import List, Dict, Tuple, Union, Optional

def merge_chunks(
    segments: SlidingWindowFeature,
    chunk_size: float,
    onset: float = 0.5,
    offset: Optional[float] = None,
) -> List[Dict[str, Union[float, List[Tuple[float, float]]]]]:
    """
    Merge operation described in paper
    """
    curr_end: float = 0.0
    merged_segments: List[Dict[str, Union[float, List[Tuple[float, float]]]]] = []
    seg_idxs: List[Tuple[float, float]] = []
    speaker_idxs: List[str] = []

    assert chunk_size > 0
    binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
    segments = binarize(segments)
    segments_list: List[SegmentX] = []
    for speech_turn in segments.get_timeline():
        segments_list.append(SegmentX(speech_turn.start, speech_turn.end, "UNKNOWN"))

    if len(segments_list) == 0:
        return []

    curr_start = segments_list[0].start

    for seg in segments_list:
        if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
            merged_segments.append({
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,  # Now this is allowed
            })
            curr_start = seg.start
            seg_idxs = []
            speaker_idxs = []
        curr_end = seg.end
        seg_idxs.append((seg.start, seg.end))
        speaker_idxs.append(seg.speaker)

    # Add the final segment
    merged_segments.append({ 
        "start": curr_start,
        "end": curr_end,
        "segments": seg_idxs,
    })

    return merged_segments


class VAD(Module):
    def __init__(self,
                    device: str = "cuda",
                    model_path: Optional[str] = ".models/vad-whisperx",
                    max_chunk_size: float = 30.0,
                    last_time_spoken_offset: float = 3,
                    vad_onset: float = 0.500,
                    vad_offset: float = 0.363,
                    use_auth_token: Union[Text, None] = None,
                    model_fp: Union[Text, None] = None,
                    vad_segmentation_url: str = "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
                ) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="VAD-Module"
        )
        self.device: str = device if torch.cuda.is_available() else "cpu"
        self.model_path: Optional[str] = model_path
        self.max_chunk_size: float = max_chunk_size
        self.last_time_spoken_offset: float = last_time_spoken_offset # It will stop processing if no one has spoken in the last 5 seconds
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.use_auth_token=use_auth_token
        self.model_fp=model_fp
        self.vad_segmentation_url = vad_segmentation_url

        self._model: Optional[VoiceActivitySegmentation] = None

    def init_module(self) -> None:
        log.info("Loading model vad...")
        self._model = self.load_vad_model(device=self.device, vad_onset=self.vad_onset, vad_offset=self.vad_offset, use_auth_token=self.use_auth_token, model_fp=self.model_fp)
        log.info("VAD model loaded")

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self._model:
            raise Exception("Whisper model not loaded")
        if not dp.data:
            raise Exception("No data found")
        if dp.data.audio_data is None:
            raise Exception("No audio data found")
        if dp.data.audio_buffer_time is None:
            raise Exception("No audio buffer time found")
        if dp.data.audio_data_sample_rate is None:
            raise Exception("No sample rate found")

        # sample_rate: int = dp.data.audio_data_sample_rate
        audio_time: float = dp.data.audio_buffer_time
        audio_chunk: float = dp.data.audio_buffer_time if dp.data.audio_buffer_time < self.max_chunk_size else self.max_chunk_size
        audio: np.ndarray = dp.data.audio_data

        # Perform voice activity detection
        vad_result: SlidingWindowFeature = self._model.apply(audio, sr=dp.data.audio_data_sample_rate)

        # Merge VAD segments if necessary
        merged_segments: List[Dict[str, float | List[Tuple[float, float]]]] = merge_chunks(vad_result, chunk_size=audio_chunk)

        dp.data.vad_result = merged_segments

        last_time_spoken: float = 0.0
        if len(merged_segments) > 0:
            # detect if someone has spoken in the last 5 seconds
            last_segment = merged_segments[-1]
            if type(last_segment['end']) == float:
                last_time_spoken = last_segment['end']

        if len(merged_segments) == 0 or last_time_spoken < (audio_time - self.last_time_spoken_offset):
            dpm.message = "No voice detected"
            dpm.status = Status.EXIT
            print("No active speech found in audio")
            return


    def load_vad_model(self, device: str, vad_onset: float=0.500, vad_offset: float=0.363, use_auth_token: Union[Text, None]=None, model_fp: Union[Text, None]=None) -> VoiceActivitySegmentation:
        model_dir = self.model_path if self.model_path is not None else torch.hub._get_torch_home()
        os.makedirs(model_dir, exist_ok = True)
        if model_fp is None:
            model_fp = os.path.join(model_dir, "whisperx-vad-segmentation.bin")
        if os.path.exists(model_fp) and not os.path.isfile(model_fp):
            raise RuntimeError(f"{model_fp} exists and is not a regular file")

        if not os.path.isfile(model_fp):
            with urllib.request.urlopen(self.vad_segmentation_url) as source, open(model_fp, "wb") as output:
                with tqdm(
                    total=int(source.info().get("Content-Length")),
                    ncols=80,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))

        model_bytes = open(model_fp, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() != self.vad_segmentation_url.split('/')[-2]:
            raise RuntimeError(
                "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
            )

        vad_model = Model.from_pretrained(model_fp, use_auth_token=use_auth_token)
        hyperparameters = {"onset": vad_onset, 
                        "offset": vad_offset,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1}
        vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
        vad_pipeline.instantiate(hyperparameters)

        return vad_pipeline

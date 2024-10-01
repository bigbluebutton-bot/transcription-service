# main.py
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import pickle
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Tuple
from urllib.parse import quote, urlencode, urlunparse
from prometheus_client import start_http_server
import copy

import requests # type: ignore
from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

from m_convert_audio import Convert_Audio
from m_create_audio_buffer import Create_Audio_Buffer
from m_faster_whisper import Faster_Whisper_transcribe
from m_confirm_words import Confirm_Words
from m_rate_limiter import Rate_Limiter
from m_vad import VAD
import data
import logger
from simulate_live_audio_stream import Statistics, simulate_live_audio_stream, stats, transcribe_audio

log = logger.setup_logging()

start_http_server(8042)

faster_whisper_model_path: str = ".models/faster-whisper"





@dataclass
class Prometheus_URL:
    scheme: str
    netloc: str
    path: str
    query: Dict[str, str] = field(default_factory=dict)
    params: str = ''
    fragment: str = ''

    def __str__(self):
        # Properly encode the query parameters without iterating over characters
        encoded_query = urlencode(self.query)
        # Construct the URL using urlunparse
        return urlunparse((self.scheme, self.netloc, self.path, self.params, encoded_query, self.fragment))

    def copy(self):
        # Return a shallow copy of the instance
        return copy.copy(self)


@dataclass
class Simulation_Pipeline:
    name: str
    prometheus_url: List[Prometheus_URL]
    pipeline: Pipeline

# CreateNsAudioPackage, Load_audio, VAD, Faster_Whisper_transcribe, Local_Agreement
scheme = "http"
netloc = "prometheus-to-graph:5000"
graph_server = "http://prometheus:9090"

simulation_pipeline_list = [
    Simulation_Pipeline(
        name = "1-10s-2flowrate-batching-2confirm",
        prometheus_url = [

                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/stats",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "label": "module_name|pipeline_name",
                    }
                ),

                # Processing Time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/graph",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "title": "Verarbeitungszeit 10s AudioBuffer",
                        "xlabel": "Zeit",
                        "ylabel": "Verarbeitungszeit in Sekunden",
                        "legend": "true",
                        "label": "module_name|pipeline_name",
                    }
                ),
                
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/stats",
                    query={
                        "server": graph_server,
                        "query": "rate(pipeline_input_flowrate_total{pipeline_id=\"PIPELINEID\"}[2s])|rate(module_exit_flowrate_total{pipeline_id=\"PIPELINEID\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "label": "pipeline_name|module_name",
                    }
                ),
                
                # Exit flowrate of each model
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/graph",
                    query={
                        "server": graph_server,
                        "query": "rate(pipeline_input_flowrate_total{pipeline_id=\"PIPELINEID\"}[2s])|rate(module_exit_flowrate_total{pipeline_id=\"PIPELINEID\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "title": "Exit-Flowrate",
                        "xlabel": "Zeit",
                        "ylabel": "Flowrate in Datenpackete pro Sekunde",
                        "legend": "true",
                        "label": "pipeline_name|module_name",
                    }
                ),

                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/stats",
                    query={
                        "server": graph_server,
                        "query": "rate(pipeline_output_flowrate_total{pipeline_id=\"PIPELINEID\"}[2s])|rate(module_exit_flowrate_total{pipeline_id=\"PIPELINEID\", module_name=\"VAD-Module\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "label": "pipeline_name|module_name",
                    }
                ),

                # Flowrate output + VAD exit
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/graph",
                    query={
                        "server": graph_server,
                        "query": "rate(pipeline_output_flowrate_total{pipeline_id=\"PIPELINEID\"}[2s])|rate(module_exit_flowrate_total{pipeline_id=\"PIPELINEID\", module_name=\"VAD-Module\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "title": "VAD",
                        "xlabel": "Zeit",
                        "ylabel": "Flowrate in Datenpackete pro Sekunde",
                        "legend": "true",
                        "label": "pipeline_name|module_name",
                    }
                ),
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=10,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=10,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=2.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],
                                            
        ),
    ),
    
    


    Simulation_Pipeline(
        name = "2-10s-2flowrate-nobatching-2confirm",
        prometheus_url = [
            
                # stats processing time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/stats",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "label": "module_name|pipeline_name",
                    }
                ),
            
                # Processing Time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/graph",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[2s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[2s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "title": "Verarbeitungszeit 10s AudioBuffer, kein Batching",
                        "xlabel": "Zeit",
                        "ylabel": "Verarbeitungszeit in Sekunden",
                        "legend": "true",
                        "label": "module_name|pipeline_name",
                    }
                ),
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=10,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=10,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=False,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=2.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),


    
    
    Simulation_Pipeline(
        name = "3-30s-2flowrate-batching-2confirm",
        prometheus_url = [
            
                # stats processing time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/stats",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "label": "module_name|pipeline_name",
                    }
                ),
            
                # Processing Time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/graph",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "title": "Verarbeitungszeit 30s AudioBuffer",
                        "xlabel": "Zeit",
                        "ylabel": "Verarbeitungszeit in Sekunden",
                        "legend": "true",
                        "label": "module_name|pipeline_name",
                    }
                ),
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=30,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=30,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=2.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),



    Simulation_Pipeline(
        name = "4-30s-2flowrate-nobatching-2confirm",
        prometheus_url = [
            
                # stats processing time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/stats",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "label": "module_name|pipeline_name",
                    }
                ),
            
                # Processing Time
                Prometheus_URL(
                    scheme=scheme,
                    netloc=netloc,
                    path="/graph",
                    query={
                        "server": graph_server,
                        "query": "rate(module_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(module_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])|rate(pipeline_success_time_sum{pipeline_id=\"PIPELINEID\"}[3s]) / rate(pipeline_success_time_count{pipeline_id=\"PIPELINEID\"}[3s])",
                        "start": "STARTTIME",
                        "end": "ENDTIME",
                        "title": "Verarbeitungszeit 30s AudioBuffer",
                        "xlabel": "Zeit",
                        "ylabel": "Verarbeitungszeit in Sekunden",
                        "legend": "true",
                        "label": "module_name|pipeline_name",
                    }
                ),
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=30,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=30,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=False,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=2.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        
        
        
        



    Simulation_Pipeline(
        name = "5-10s-2flowrate-batching-0confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=10,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=10,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=10,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=0.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        
        



    Simulation_Pipeline(
        name = "6-10s-2flowrate-batching-1confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=10,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=10,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=1.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        
        



    Simulation_Pipeline(
        name = "7-10s-2flowrate-batching-2confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=10,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=10,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=2.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        




    Simulation_Pipeline(
        name = "8-10s-2flowrate-batching-3confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=10,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=10,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=3.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),











    Simulation_Pipeline(
        name = "5-30s-2flowrate-batching-0confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=30,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=10,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=30,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=0.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        
        



    Simulation_Pipeline(
        name = "6-30s-2flowrate-batching-1confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=30,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=30,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=1.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        
        



    Simulation_Pipeline(
        name = "7-30s-2flowrate-batching-2confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=30,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=30,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=2.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
        
        




    Simulation_Pipeline(
        name = "8-30s-2flowrate-batching-3confirm",
        prometheus_url = [
            ],
        pipeline = Pipeline[data.AudioData](name="Pipeline", controllers_or_phases=[
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    queue_size=10,
                    name="Create_Audio_Buffer",
                    phases=[
                        PipelinePhase(
                            name="Create_Audio_Buffer",
                            modules=[
                                Create_Audio_Buffer(
                                        last_n_seconds=30,
                                    ),
                                Rate_Limiter(
                                        flowrate_per_second=2,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=3,
                    queue_size=2,
                    name="AudioPreprocessingController",
                    phases=[
                        PipelinePhase(
                            name="VADPhase",
                            modules=[
                                Convert_Audio(),
                                VAD(
                                        max_chunk_size=30,
                                        last_time_spoken_offset=3.0,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.FIRST_WINS,
                    max_workers=1,
                    queue_size=0,
                    name="MainProcessingController",
                    phases=[
                        PipelinePhase(
                            name="WhisperPhase",
                            modules=[
                                Faster_Whisper_transcribe(
                                        model_size="large-v3",
                                        task="transcribe",
                                        compute_type="float16",
                                        batching=True,
                                        batch_size=32,
                                    ),
                            ]
                        )
                    ]
                ),
                PipelineController(
                    mode=ControllerMode.NOT_PARALLEL,
                    max_workers=1,
                    name="OutputController",
                    phases=[
                        PipelinePhase(
                            name="OutputPhase",
                            modules=[
                                Confirm_Words(
                                        max_confirmed_words=0,
                                        confirm_if_older_then=3.0,
                                    ),
                            ]
                        )
                    ]
                )
            ],         
        ),
    ),
]




result: List[DataPackage[data.AudioData]] = []
result_mutex = threading.Lock()
def callback(dp: DataPackage[data.AudioData]) -> None:
    if dp.data and dp.data.transcribed_segments:
        with result_mutex:
            result.append(dp)
            
        processing_time = dp.total_time

        if dp.data.confirmed_words is not None:
            only_words_c = [w.word for w in dp.data.confirmed_words]

        if dp.data.unconfirmed_words is not None:
            only_words_u = [w.word for w in dp.data.unconfirmed_words]
            
        if len(only_words_c) > 50:
            only_words_c = only_words_c[-50:]
            
        # print(f"({new_words}){only_words_c} ++ {only_words_u}")
        print(f"{processing_time:2f}:  {only_words_c} ")
    pass

def error_callback(dp: DataPackage[data.AudioData]) -> None:
    log.error("Pipeline error", extra={"data_package": dp})

audio_extensions = [
    ".aac",    # Advanced Audio Codec
    ".ac3",    # Audio Codec 3
    ".aiff",   # Audio Interchange File Format
    ".aif",    # Audio Interchange File Format
    ".alac",   # Apple Lossless Audio Codec
    ".amr",    # Adaptive Multi-Rate audio codec
    ".ape",    # Monkey's Audio
    ".au",     # Sun Microsystems Audio
    ".dts",    # Digital Theater Systems audio
    ".eac3",   # Enhanced AC-3
    ".flac",   # Free Lossless Audio Codec
    ".m4a",    # MPEG-4 Audio (usually AAC)
    ".mka",    # Matroska Audio
    ".mp3",    # MPEG Layer 3
    ".ogg",    # Ogg Vorbis or Ogg Opus
    ".opus",   # Opus audio codec
    ".ra",     # RealAudio
    ".rm",     # RealMedia
    ".tta",    # True Audio codec
    ".voc",    # Creative Voice File
    ".wav",    # Waveform Audio File Format
    ".wma",    # Windows Media Audio
    ".wv",     # WavPack
    ".caf",    # Core Audio Format
    ".gsm",    # GSM 6.10 audio codec
    ".mp2",    # MPEG Layer 2 audio
    ".spx",    # Speex audio
    ".aob"     # Audio Object (used in DVD-Audio)
]

def main() -> None:
    input_folder  = './audio'
    output_folder = './simulate_results'
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder_name in os.listdir(input_folder):
        # Check if the file has a valid audio extension and skip non-audio files
        if not any(folder_name.endswith(ext) for ext in audio_extensions):
            print(f"Skipping non-audio file: {folder_name}")
            continue
        
        new_output_folder = os.path.join(output_folder, os.path.splitext(folder_name)[0])  # Create folder for each file
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        input_file = os.path.join(input_folder, folder_name)
        output_file = os.path.join(new_output_folder, os.path.splitext(folder_name)[0] + '.ogg')
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            continue

        # Construct and run the ffmpeg command as before
        command = [
            'ffmpeg', '-i', input_file, '-c:a', 'libopus',
            '-frame_duration', '20', '-page_duration', '20000',
            '-vn', output_file
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Converted: {folder_name} -> {output_file}")
        except Exception as e:
            print(f"Error processing file {folder_name}: {e}")
    
    for folder_name in os.listdir(output_folder):
        new_output_folder = os.path.join(output_folder, os.path.splitext(folder_name)[0])
        file_path = os.path.join(new_output_folder, folder_name + ".ogg")
        new_file_beginning = os.path.join(new_output_folder, folder_name)
        
        if not file_path.endswith(".ogg"):
            print(f"Skipping non-audio file: {folder_name}")
            continue
        

        for simulation_pipeline in simulation_pipeline_list:
            # clear data
            with result_mutex:
                result.clear()
            
            new_file_beginning_sumulation = new_file_beginning + f"_{simulation_pipeline.name}"
            
            start_time = end_time = time.time()
            if not os.path.exists(f"{new_file_beginning_sumulation}_simulation.pkl"):
                # create pipeline
                pipeline = simulation_pipeline.pipeline
                pipeline_id = pipeline.get_id()
                print(f"Pipeline ID: {pipeline_id}")
                instance = pipeline.register_instance()

                def simulated_callback(raw_audio_data: bytes) -> None:
                    audio_data = data.AudioData(raw_audio_data=raw_audio_data)
                    pipeline.execute(
                                    audio_data, instance, 
                                    callback=callback,
                                    error_callback=error_callback
                                    )
                
                start_time = time.time()
                simulate_live_audio_stream(file_path, simulated_callback)
                end_time = time.time()
                time.sleep(5)
                
                pipeline.unregister_instance(instance)
                del pipeline

                with result_mutex:
                    data_list = [(dat.data, dat.start_time, dat.end_time) for dat in result if dat.data is not None]
                    with open(f"{new_file_beginning_sumulation}_simulation.pkl", 'wb') as file:
                        pickle.dump(data_list, file)
                        
                        
                # Save graphes from url
                start = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
                end = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
                urls = simulation_pipeline.prometheus_url
                for i, url in enumerate(urls):
                    if os.path.exists(f"{new_file_beginning_sumulation}_{i}_graph.png"):
                        continue
                    
                    url_copy = url.copy()
                    
                    if url_copy.path == "/graph":
                        url_copy.query["query"] = url_copy.query["query"].replace("PIPELINEID", pipeline_id)
                        url_copy.query["start"] = start
                        url_copy.query["end"] = end
                        print(url_copy)
                    
                        response = requests.get(url_copy)

                        # Check if the request was successful
                        if response.status_code == 200:
                            # Save the content as an image file
                            with open(f"{new_file_beginning_sumulation}_{i}_graph.png", 'wb') as file:
                                file.write(response.content)
                        else:
                            print(f"Failed to download image. Status code: {response.status_code}")
                            
                    elif url_copy.path == "/stats":
                        url_copy.query["query"] = url_copy.query["query"].replace("PIPELINEID", pipeline_id)
                        url_copy.query["start"] = start
                        url_copy.query["end"] = end
                        print(url_copy)
                    
                        statsresponse = requests.get(url_copy)

                        if statsresponse.status_code == 200:
                            try:
                                # Attempt to parse the content as JSON
                                response_data = statsresponse.json()
                                
                                # Save the content as a JSON file
                                with open(f"{new_file_beginning_sumulation}_{i}_stats.json", 'w') as statsfile:
                                    import json
                                    json.dump(response_data, statsfile, indent=4)
                            except ValueError:
                                print("Failed to parse the response as JSON.")
                        else:
                            print(f"Failed to download data. Status code: {statsresponse.status_code}")
                        

            if not os.path.exists(f"{new_file_beginning}_transcript.pkl"):
                transcript = transcribe_audio(file_path, faster_whisper_model_path)
                with open(f"{new_file_beginning}_transcript.pkl", 'wb') as file:
                    pickle.dump(transcript, file)



            # Load the pkl file
            with open(f"{new_file_beginning_sumulation}_simulation.pkl", 'rb') as read_file:
                live_data: list[tuple[data.AudioData, float, float]] = pickle.load(read_file) # type: ignore
                
            with open(f"{new_file_beginning}_transcript.pkl", 'rb') as read_file:
                transcript_words: List[data.Word] = pickle.load(read_file) # type: ignore

            live_dps: List[DataPackage[data.AudioData]] = [] # type: ignore
            for da in live_data:
                new_dp = DataPackage[data.AudioData]()
                new_dp.data=da[0]
                new_dp.start_time=da[1]
                new_dp.end_time=da[2]
                live_dps.append(new_dp)

            # cw = Confirm_Words()
            # for live_dp in live_dps:
            #     if live_dp.data is not None:
            #         live_dp.data.confirmed_words = None
            #         live_dp.data.unconfirmed_words = None
            #     cw.execute(live_dp, DataPackageController(), DataPackagePhase(), DataPackageModule())

            stat_sensetive, stat_insensetive, avg_time_difference, std_dev, mad = stats(live_dps, transcript_words)
            
            def save_stats(stats_sensetive, stats_insensetive, stat_avg_time_difference, stat_std_dev, stat_mad) -> None:
                # Function to format statistics as JSON
                def stats_to_json(stat: Statistics) -> str:
                    return json.dumps({
                        "deletions": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in stat.deletions],
                        "substitutions": [{"from": sub[0].word, "to": sub[1].word} for sub in stat.substitutions],
                        "insertions": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in stat.insertions],
                        "wer": stat.wer,
                        "avg_delta_start": stat.avg_delta_start,
                        "avg_delta_end": stat.avg_delta_end
                    }, indent=4)

                # if file f"{new_file_beginning}_stats.txt" exists, delete it
                if os.path.exists(f"{new_file_beginning_sumulation}_stats.txt"):
                    os.remove(f"{new_file_beginning_sumulation}_stats.txt")

                # Writing the output to a file
                with open(f"{new_file_beginning_sumulation}_stats.txt", "w") as file:
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"File: {file_path}\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Average time difference between live and transcript: {stat_avg_time_difference * 1000:.1f} milliseconds\n")
                    file.write(f"Standard deviation of time difference: {stat_std_dev * 1000:.1f} milliseconds\n")
                    file.write(f"Mean absolute deviation of time difference: {stat_mad * 1000:.1f} milliseconds\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Statistics for case sensitive:\n")
                    file.write(f"Number of words missing in live (Deletions): {len(stats_sensetive.deletions)}\n")
                    file.write(f"Number of wrong words in live (Substitutions): {len(stats_sensetive.substitutions)}\n")
                    file.write(f"Number of extra words in live (Insertions): {len(stats_sensetive.insertions)}\n")
                    file.write(f"Average difference in start times: {stats_sensetive.avg_delta_start * 1000:.1f} milliseconds\n")
                    file.write(f"Average difference in end times: {stats_sensetive.avg_delta_end * 1000:.1f} milliseconds\n")
                    file.write(f"Word Error Rate (WER): {stats_sensetive.wer * 100:.1f}%\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Statistics without case sensitivity and symbols:\n")
                    file.write(f"Number of words missing in live (Deletions): {len(stats_insensetive.deletions)}\n")
                    file.write(f"Number of wrong words in live (Substitutions): {len(stats_insensetive.substitutions)}\n")
                    file.write(f"Number of extra words in live (Insertions): {len(stats_insensetive.insertions)}\n")
                    file.write(f"Average difference in start times: {stats_insensetive.avg_delta_start * 1000:.1f} milliseconds\n")
                    file.write(f"Average difference in end times: {stats_insensetive.avg_delta_end * 1000:.1f} milliseconds\n")
                    file.write(f"Word Error Rate (WER): {stats_insensetive.wer * 100:.1f}%\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Statistics as formatted JSON for sensitive case:\n")
                    file.write(stats_to_json(stats_sensetive) + "\n")
                    file.write(f"Statistics as formatted JSON for insensitive case:\n")
                    file.write(stats_to_json(stats_insensetive) + "\n")
                
            print(f"File: {folder_name}")
            save_stats(stat_sensetive, stat_insensetive, avg_time_difference, std_dev, mad)
            
            try:
                subprocess.run(['chmod', '777', output_folder, '-R'])
            except Exception as e:
                pass
    
if __name__ == "__main__":
    main()
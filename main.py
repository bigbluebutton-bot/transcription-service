# main.py
import threading
import time
from typing import Dict, List, Tuple

from prometheus_client import start_http_server
from flask import Flask

from stream_pipeline.data_package import DataPackage
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

from Config import load_settings
from StreamServer import Server, Client as StreamClient
from m_convert_audio import Convert_Audio
from m_create_audio_buffer import Create_Audio_Buffer
from m_faster_whisper import Faster_Whisper_transcribe, WhisperAIManager
from m_confirm_words import Confirm_Words
from m_rate_limiter import Rate_Limiter
from m_vad import VAD
import data
import logger

log = logger.setup_logging()

start_http_server(8042)

settings = load_settings()

# load whisper singelton
WhisperAIManager(
    model_path = settings["FASTER_WHISPER_MODEL_PATH"],
    model_size = settings["FASTER_WHISPER_MODEL_SIZE"], #tiny, tiny.en, small, small.en, base, base.en, medium, medium.en, large-v1, large-v2, large-v3
    compute_type = settings["FASTER_WHISPER_COMPUTE_TYPE"], # "float16" or "int8"
    batching = settings["FASTER_WHISPER_BATCHING"],
    batch_size = settings["FASTER_WHISPER_BATCH_SIZE"],
    devices = settings["FASTER_WHISPER_DEVICE"] # "cuda" or "cpu"
)

controllers = [
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
                        last_n_seconds=settings["AUDIO_BUFFER_LAST_N_SECONDS"],
                        min_n_seconds=settings["AUDIO_BUFFER_MIN_N_SECONDS"]
                    ),
                    Rate_Limiter(
                        flowrate_per_second=settings["FLOWRATE_PER_SECOND"]
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
                    Convert_Audio(
                        convert_sample_rate=settings["CONVERT_SAMPLE_RATE"]
                    ),
                    VAD(
                        device = settings["VAD_DEVICE"],
                        model_path = settings["VAD_MODEL_PATH"],
                        max_chunk_size = settings["VAD_MAX_CHUNK_SIZE"],
                        last_time_spoken_offset = settings["VAD_LAST_TIME_SPOKEN_OFFSET"],
                        vad_onset = settings["VAD_ONSET"],
                        vad_offset = settings["VAD_OFFSET"],
                        use_auth_token = None if settings["VAD_USE_AUTH_TOKEN"] == "" else settings["VAD_USE_AUTH_TOKEN"],
                        model_fp = None if settings["VAD_MODEL_FP"] == "" else settings["VAD_MODEL_FP"],
                        vad_segmentation_url = settings["VAD_SEGMENTATION_URL"]
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
                        task = settings["FASTER_WHISPER_TASK"], # transcribe, translate
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
                        offset = settings["CONFIRM_WORDS_OFFSET"],
                        max_confirmed_words=settings["CONFIRM_WORDS_MAX_WORDS"],
                        confirm_if_older_then=settings["CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN"]
                    ),
                ]
            )
        ]
    )
]

pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")

# Health check http sever
app = Flask(__name__)
STATUS = "stopped" # starting, running, stopping, stopped
@app.route('/health', methods=['GET'])
def healthcheck() -> Tuple[str, int]:
    global STATUS
    print(STATUS)
    if STATUS == "running":
        return STATUS, 200
    else:
        return STATUS, 503

def main() -> None:
    global STATUS
    STATUS = "starting"

    # Start the health http-server (flask) in a new thread.
    webserverthread = threading.Thread(target=app.run, kwargs={'debug': False, 'host': settings["HOST"], 'port': settings["HEALTH_CHECK_PORT"]})
    webserverthread.daemon = True  # This will ensure the thread stops when the main thread exits
    webserverthread.start()

    client_dict: Dict[str, StreamClient] = {}   # Dictionary with all connected clients (key: instance_id, value: StreamClient)
    client_dict_mutex = threading.Lock()        # Mutex to lock the client_dict

    # Pipeline callbacks
    def callback(dp: DataPackage[data.AudioData]) -> None:
        if dp.data and dp.data.confirmed_words is not None and dp.data.unconfirmed_words is not None:
            # log.info(f"Text: {dp.data.transcribed_text['words']}")
            processing_time = dp.total_time
            # log.info(f"{processing_time:2f}:  {dp.data.confirmed_words} +++ {dp.data.unconfirmed_words}")
            # log.info(f"{processing_time:2f}: cleaned_words:  {dp.data.transcribed_segments}")

            
            # put dp.data.confirmed_words together with space
            only_words: List[data.Word] = []
            for word in dp.data.confirmed_words:
                only_words.append(word)
            
            text = ""
            for word in only_words:
                # if there is a . in this word add \n behind it
                # if "." in word.word:
                #     text += word.word + "\n"
                # else:
                text += word.word + " "
            # for word in dp.data.unconfirmed_words:
            #     text += word.word + " "
            
            # get client
            instance_id = dp.pipeline_instance_id
            print(f"Instance: {instance_id}")
            with client_dict_mutex:
                if not instance_id in client_dict:
                    log.error(f"Instance {instance_id} not in client_dict!")
                    pipeline.unregister_instance(instance_id)
                    return
                
                # send text to client
                client = client_dict[instance_id]
                client.send_message(str.encode(text))

    
    def exit_callback(dp: DataPackage[data.AudioData]) -> None:
        # log.info(f"Exit: {dp.controllers[-1].phases[-1].modules[-1].message}")
        pass

    def overflow_callback(dp: DataPackage[data.AudioData]) -> None:
        # log.info("Overflow")
        pass

    def outdated_callback(dp: DataPackage[data.AudioData]) -> None:
        log.info("Outdated", extra={"data_package": dp})

    def error_callback(dp: DataPackage[data.AudioData]) -> None:
        log.error("Pipeline error", extra={"data_package": dp})

    # Create server
    host = str(settings["HOST"])
    tcp_port = int(str(settings["TCPPORT"]))
    udp_port = int(str(settings["UDPPORT"]))
    secret_token = str(settings["SECRET_TOKEN"])
    external_host = str(settings["EXTERNALHOST"])
    srv = Server(host, tcp_port, udp_port, secret_token, 4096, 5, 10, 1024, external_host)

    # Handle new connections and disconnections, timeouts and messages
    def OnConnected(c: StreamClient) -> None:
        print(f"Connected by {c.tcp_address()}")

        # Create new client
        new_instance = pipeline.register_instance()
        
        with client_dict_mutex:
            client_dict[new_instance] = c

        # Handle disconnections
        def ondisconnedted(c: StreamClient) -> None:
            print(f"Disconnected by {c.tcp_address()}")
            # Remove client from client_dict
            with client_dict_mutex:
                if c in client_dict.values():
                    instance_id = [key for key, value in client_dict.items() if value == c][0]
                    pipeline.unregister_instance(instance_id)
                    del client_dict[instance_id]
        c.on_disconnected(ondisconnedted)

        # Handle timeouts
        def ontimeout(c: StreamClient) -> None:
            print(f"Timeout by {c.tcp_address()}")
            # Remove client from client_dict
            with client_dict_mutex:
                if c in client_dict.values():
                    instance_id = [key for key, value in client_dict.items() if value == c][0]
                    pipeline.unregister_instance(instance_id)
                    del client_dict[instance_id]
        c.on_timeout(ontimeout)

        # Handle messages
        def onmsg(c: StreamClient, recv_data: bytes) -> None:
            # print(f"UDP from: {c.tcp_address()}")
            with client_dict_mutex:
                if not c in client_dict.values():
                    print(f"Client not in client_dict")
                    c.stop()
                    return


                instance_id = [key for key, value in client_dict.items() if value == c][0]
            
                audio_data = data.AudioData(raw_audio_data=recv_data)
                pipeline.execute(
                                audio_data, 
                                instance_id=instance_id, 
                                callback=callback, 
                                exit_callback=exit_callback, 
                                overflow_callback=overflow_callback, 
                                outdated_callback=outdated_callback, 
                                error_callback=error_callback
                                )
                

        c.on_udp_message(onmsg)
    srv.on_connected(OnConnected)

    # Start server
    print(f"Starting server: {settings['HOST']}:{settings['TCPPORT']}...")
    srv.start()
    print("Ready to transcribe. Press Ctrl+C to stop.")

    STATUS = "running"

    # Wait until stopped by Strg + C
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass

    # Stop server
    STATUS = "stopping"
    print("Stopping server...")
    srv.stop()
    print("Server stopped")
    STATUS = "stopped"
    
if __name__ == "__main__":
    main()
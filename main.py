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
from Client import Client
from m_convert_audio import Convert_Audio
from m_create_audio_buffer import Create_Audio_Buffer
from m_faster_whisper import Faster_Whisper_transcribe
from m_confirm_words import Confirm_Words
from m_rate_limiter import Rate_Limiter
from m_vad import VAD
import data
import logger

log = logger.setup_logging()

start_http_server(8042)

# CreateNsAudioPackage, Load_audio, VAD, Faster_Whisper_transcribe, Local_Agreement
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
                        last_n_seconds=30
                    ),
                    Rate_Limiter(),
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
                    VAD(),
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
                    Faster_Whisper_transcribe(),
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
                        confirm_if_older_then=1.0,
                        max_confirmed_words=50
                    ),
                ]
            )
        ]
    )
]

pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")

instance = pipeline.register_instance()

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
    
    settings = load_settings()

    # Start the health http-server (flask) in a new thread.
    webserverthread = threading.Thread(target=app.run, kwargs={'debug': False, 'host': settings["HOST"], 'port': settings["HEALTH_CHECK_PORT"]})
    webserverthread.daemon = True  # This will ensure the thread stops when the main thread exits
    webserverthread.start()

    client_dict: Dict[StreamClient, Client] = {}        # Dictionary with all connected clients
    client_dict_mutex = threading.Lock() # Mutex to lock the client_dict

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
            with client_dict_mutex:
                for c in client_dict:
                    client = client_dict[c]
                    if client._instance == dp.pipeline_instance_id:
                        client.send(str.encode(text))
    
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
    tcp_port = int(settings["TCPPORT"])
    udp_port = int(settings["UDPPORT"])
    secret_token = str(settings["SECRET_TOKEN"])
    external_host = str(settings["EXTERNALHOST"])
    srv = Server(host, tcp_port, udp_port, secret_token, 4096, 5, 10, 1024, external_host)

    # Handle new connections and disconnections, timeouts and messages
    def OnConnected(c: StreamClient) -> None:
        print(f"Connected by {c.tcp_address()}")

        # Create new client
        newclient = Client(c)
        newclient._instance = instance # TODO
        # newclient._instance = pipeline.register_instance()
        with client_dict_mutex:
            client_dict[c] = newclient

        # Handle disconnections
        def ondisconnedted(c: StreamClient) -> None:
            print(f"Disconnected by {c.tcp_address()}")
            # Remove client from client_dict
            with client_dict_mutex:
                if c in client_dict:
                    del client_dict[c]
        c.on_disconnected(ondisconnedted)

        # Handle timeouts
        def ontimeout(c: StreamClient) -> None:
            print(f"Timeout by {c.tcp_address()}")
            # Remove client from client_dict
            with client_dict_mutex:
                if c in client_dict:
                    del client_dict[c]
        c.on_timeout(ontimeout)

        # Handle messages
        def onmsg(c: StreamClient, recv_data: bytes) -> None:
            # print(f"UDP from: {c.tcp_address()}")
            with client_dict_mutex:
                if not c in client_dict:
                    print(f"Client {c.tcp_address()} not in list!")
                    return
                client = client_dict[c]
                
                if client._instance is None:
                    print(f"Client {c.tcp_address()} has no instance!")
                    return
            
                audio_data = data.AudioData(raw_audio_data=recv_data)
                pipeline.execute(
                                audio_data, 
                                client._instance, 
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
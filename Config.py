import os
import sys
import logging
from dotenv import load_dotenv
from typing import Callable, Dict, Optional, TypedDict, Union

class Settings(TypedDict):
    HOST: str
    EXTERNALHOST: str
    TCPPORT: int
    UDPPORT: int
    SECRET_TOKEN: str
    HEALTH_CHECK_PORT: int
    PROMETHEUS_PORT: int
    AUDIO_BUFFER_LAST_N_SECONDS: int
    AUDIO_BUFFER_MIN_N_SECONDS: int
    FLOWRATE_PER_SECOND: float
    CONVERT_SAMPLE_RATE: int
    VAD_DEVICE: str
    VAD_MODEL_PATH: str
    VAD_MAX_CHUNK_SIZE: float
    VAD_LAST_TIME_SPOKEN_OFFSET: int
    VAD_ONSET: float
    VAD_OFFSET: float
    VAD_USE_AUTH_TOKEN: str
    VAD_MODEL_FP: str
    VAD_SEGMENTATION_URL: str
    FASTER_WHISPER_MODEL_PATH: str
    FASTER_WHISPER_MODEL_SIZE: str
    FASTER_WHISPER_TASK: str
    FASTER_WHISPER_COMPUTE_TYPE: str
    FASTER_WHISPER_BATCHING: bool
    FASTER_WHISPER_BATCH_SIZE: int
    FASTER_WHISPER_DEVICE: list[str]
    CONFIRM_WORDS_OFFSET: float
    CONFIRM_WORDS_MAX_WORDS: int
    TRANSCRIBE_CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN: float
    TRANSLATE_CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN: float

def load_settings() -> Settings:
    # load_dotenv(override=True)

    valid_config: bool = True

    def validate_model(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        nonlocal valid_config
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if value not in valid_models:
            logging.error(f"Invalid MODEL setting: {env_var}. Must be one of {valid_models}.")
            valid_config = False
            return default
        return value

    def validate_path(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        nonlocal valid_config
        if not isinstance(value, str):
            logging.error(f"Invalid type for setting: {env_var}. Expected a string. Using default value: {default}")
            valid_config = False
            return default

        if os.path.exists(value):
            if os.path.isfile(value):
                # It's a file, all good
                return value
            elif os.path.isdir(value):
                # It's a directory, all good
                return value
            else:
                # Exists but is neither file nor directory
                logging.error(f"Invalid path for setting: {env_var}. Path exists but is not a file or directory. Using default value: {default}")
                valid_config = False
                return default
        else:
            try:
                os.makedirs(value)
                logging.info(f"Created missing directory for setting: {env_var} at path: {value}")
                return value
            except OSError as e:
                logging.error(f"Failed to create directory for setting: {env_var}. Error: {e}. Using default value: {default}")
                valid_config = False
                return default

    def validate_float(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        nonlocal valid_config
        try:
            return float(str(value))
        except ValueError:
            logging.error(f"Invalid float value for setting: {env_var}. Using default value: {default}")
            valid_config = False
            return default

    def validate_task(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        nonlocal valid_config
        valid_tasks = ['transcribe', 'translate']
        if value not in valid_tasks:
            logging.error(f"Invalid TASK setting: {env_var}. Must be one of {valid_tasks}.")
            valid_config = False
            return default
        return value

    def validate_int(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        nonlocal valid_config
        try:
            return int(str(value))
        except ValueError:
            logging.error(f"Invalid integer value for setting: {env_var}. Using default value: {default}")
            valid_config = False
            return default

    def validate_bool(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        nonlocal valid_config
        if type(value) != str and type(value) != bool:
            logging.error(f"Invalid boolean value for setting: {env_var}. Expected 'true' or 'false'. Using default value: {default}")
            valid_config = False
            return default

        if type(value) == str:
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            else:
                logging.error(f"Invalid boolean value for setting: {env_var}. Expected 'true' or 'false'. Using default value: {default}")
                valid_config = False
                return default
        else:
            if value:
                return True
            else:
                return False

    def validate_list_string(value: Union[str, int, float, bool, None, list[str]], default: Union[str, int, float, bool, None, list[str]], env_var: str) -> Union[str, int, float, bool, None, list[str]]:
        # value can be a list or a string separated by a comma which has to convert to a list
        nonlocal valid_config
        if type(value) == list:
            return value
        elif type(value) == str:
            return value.split(",")
        else:
            logging.error(f"Invalid list value for setting: {env_var}. Expected a list of strings or a comma separated string. Using default value: {default}")
            valid_config = False
            return default


    def get_variable(env_var: str, default: Union[str, int, float, bool, None, list[str]], validate_func: Optional[Callable[[Union[str, int, float, bool, None, list[str]], Union[str, int, float, bool, None, list[str]], str], Union[str, int, float, bool, None, list[str]]]] = None) -> Union[str, int, float, bool, None, list[str]]:
        value: Union[str, int, float, bool, None, list[str]] = os.getenv(env_var, default)
        if value == "None":
            value = None
        if validate_func:
            value = validate_func(value, default, env_var)
        return value

    settings: Settings = {
        'HOST': get_variable('TRANSCRIPTION_SERVER_HOST', "0.0.0.0"), # type: ignore
        'EXTERNALHOST': get_variable('TRANSCRIPTION_SERVER_EXTERNAL_HOST', "127.0.0.1"), # type: ignore
        'TCPPORT': get_variable('TRANSCRIPTION_SERVER_PORT_TCP', 5000, validate_int), # type: ignore
        'UDPPORT': get_variable('TRANSCRIPTION_SERVER_PORT_UDP', 5001, validate_int), # type: ignore
        'SECRET_TOKEN': get_variable('TRANSCRIPTION_SERVER_SECRET', "your_secret_token"), # type: ignore
        'HEALTH_CHECK_PORT': get_variable('TRANSCRIPTION_SERVER_HEALTH_CHECK_PORT', 8001, validate_int), # type: ignore
        'PROMETHEUS_PORT': get_variable('TRANSCRIPTION_SERVER_PROMETHEUS_PORT', 2112, validate_int), # type: ignore
        'AUDIO_BUFFER_LAST_N_SECONDS': get_variable('TRANSCRIPTION_AUDIO_BUFFER_LAST_N_SECONDS', 30, validate_float), # type: ignore
        'AUDIO_BUFFER_MIN_N_SECONDS': get_variable('TRANSCRIPTION_AUDIO_BUFFER_MIN_N_SECONDS', 1, validate_float), # type: ignore
        'FLOWRATE_PER_SECOND': get_variable('TRANSCRIPTION_RATE_LIMITER_FLOWRATE_PER_SECOND', 2.0, validate_float), # type: ignore
        'CONVERT_SAMPLE_RATE': get_variable('TRANSCRIPTION_CONVERT_AUDIO_CONVERT_SAMPLE_RATE', 16000, validate_int), # type: ignore
        'VAD_DEVICE': get_variable('TRANSCRIPTION_VAD_DEVICE', "cuda"), # type: ignore
        'VAD_MODEL_PATH': get_variable('TRANSCRIPTION_VAD_MODEL_PATH', ".models/vad-whisperx", validate_path), # type: ignore
        'VAD_MAX_CHUNK_SIZE': get_variable('TRANSCRIPTION_VAD_MAX_CHUNK_SIZE', 30.0, validate_float), # type: ignore
        'VAD_LAST_TIME_SPOKEN_OFFSET': get_variable('TRANSCRIPTION_VAD_LAST_TIME_SPOKEN_OFFSET', 3, validate_int), # type: ignore
        'VAD_ONSET': get_variable('TRANSCRIPTION_VAD_ONSET', 0.500, validate_float), # type: ignore
        'VAD_OFFSET': get_variable('TRANSCRIPTION_VAD_OFFSET', 0.363, validate_float), # type: ignore
        'VAD_USE_AUTH_TOKEN': get_variable('TRANSCRIPTION_VAD_USE_AUTH_TOKEN', ""), # type: ignore
        'VAD_MODEL_FP': get_variable('TRANSCRIPTION_VAD_MODEL_FP', ""), # type: ignore
        'VAD_SEGMENTATION_URL': get_variable('TRANSCRIPTION_VAD_SEGMENTATION_URL', "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"), # type: ignore
        'FASTER_WHISPER_MODEL_PATH': get_variable('TRANSCRIPTION_FASTER_WHISPER_MODEL_PATH', ".models/faster-whisper", validate_path), # type: ignore
        'FASTER_WHISPER_MODEL_SIZE': get_variable('TRANSCRIPTION_FASTER_WHISPER_MODEL_SIZE', "base"), # type: ignore
        'FASTER_WHISPER_TASK': get_variable('TRANSCRIPTION_FASTER_WHISPER_TASK', "transcribe", validate_task), # type: ignore
        'FASTER_WHISPER_COMPUTE_TYPE': get_variable('TRANSCRIPTION_FASTER_WHISPER_COMPUTE_TYPE', "float16"), # type: ignore
        'FASTER_WHISPER_BATCHING': get_variable('TRANSCRIPTION_FASTER_WHISPER_BATCHING', True, validate_bool), # type: ignore
        'FASTER_WHISPER_BATCH_SIZE': get_variable('TRANSCRIPTION_FASTER_WHISPER_BATCH_SIZE', 32, validate_int), # type: ignore
        'FASTER_WHISPER_DEVICE': get_variable('TRANSCRIPTION_FASTER_WHISPER_DEVICE', ["all"], validate_list_string), # type: ignore
        'CONFIRM_WORDS_OFFSET': get_variable('TRANSCRIPTION_CONFIRM_WORDS_OFFSET', 0.3, validate_float), # type: ignore
        'CONFIRM_WORDS_MAX_WORDS': get_variable('TRANSCRIPTION_CONFIRM_WORDS_MAX_WORDS', 50, validate_int), # type: ignore
        'TRANSCRIBE_CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN': get_variable('TRANSCRIPTION_TRANSCRIBE_CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN', 1.0, validate_float), # type: ignore
        'TRANSLATE_CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN': get_variable('TRANSCRIPTION_TRANSLATE_CONFIRM_WORDS_CONFIRM_IF_OLDER_THEN', 10.0, validate_float), # type: ignore
    }

    if not valid_config:
        logging.error("Invalid config. Please fix the errors and try again.")
        sys.exit(1)

    return settings

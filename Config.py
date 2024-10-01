import os
import sys
import logging
from dotenv import load_dotenv
from typing import Callable, Dict, Optional, Union

def load_settings() -> Dict[str, Union[str, int, float, bool]]:
    load_dotenv(override=True)

    valid_config: bool = True

    def validate_model(value: Union[str, int, float, bool], default: Union[str, int, float, bool], env_var: str) -> Union[str, int, float, bool]:
        nonlocal valid_config
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if value not in valid_models:
            logging.error(f"Invalid MODEL setting: {env_var}. Must be one of {valid_models}.")
            valid_config = False
            return default
        return value

    def validate_path(value: Union[str, int, float, bool], default: Union[str, int, float, bool], env_var: str) -> Union[str, int, float, bool]:
        nonlocal valid_config
        if type(value) != str or not os.path.exists(value):
            logging.error(f"Invalid path for setting: {env_var}. Path does not exist. Using default value: {default}")
            valid_config = False
            return default
        return value

    def validate_float(value: Union[str, int, float, bool], default: Union[str, int, float, bool], env_var: str) -> Union[str, int, float, bool]:
        nonlocal valid_config
        try:
            return float(value)
        except ValueError:
            logging.error(f"Invalid float value for setting: {env_var}. Using default value: {default}")
            valid_config = False
            return default

    def validate_task(value: Union[str, int, float, bool], default: Union[str, int, float, bool], env_var: str) -> Union[str, int, float, bool]:
        nonlocal valid_config
        valid_tasks = ['transcribe', 'translate']
        if value not in valid_tasks:
            logging.error(f"Invalid TASK setting: {env_var}. Must be one of {valid_tasks}.")
            valid_config = False
            return default
        return value

    def validate_int(value: Union[str, int, float, bool], default: Union[str, int, float, bool], env_var: str) -> Union[str, int, float, bool]:
        nonlocal valid_config
        try:
            return int(value)
        except ValueError:
            logging.error(f"Invalid integer value for setting: {env_var}. Using default value: {default}")
            valid_config = False
            return default

    def validate_bool(value: Union[str, int, float, bool], default: Union[str, int, float, bool], env_var: str) -> Union[str, int, float, bool]:
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


    def get_variable(env_var: str, default: Union[str, int, float, bool], validate_func: Optional[Callable[[Union[str, int, float, bool], Union[str, int, float, bool], str], Union[str, int, float, bool]]] = None) -> Union[str, int, float, bool]:
        value = os.getenv(env_var, default)
        if validate_func:
            value = validate_func(value, default, env_var)
        return value

    settings: Dict[str, Union[str, int, float, bool]] = {
        'HOST': get_variable('TRANSCRIPTION_SERVER_HOST', "0.0.0.0"),
        'EXTERNALHOST': get_variable('TRANSCRIPTION_SERVER_EXTERNAL_HOST', "127.0.0.1"),
        'TCPPORT': get_variable('TRANSCRIPTION_SERVER_PORT_TCP', 5000, validate_int),
        'UDPPORT': get_variable('TRANSCRIPTION_SERVER_PORT_UDP', 5001, validate_int),
        'SECRET_TOKEN': get_variable('TRANSCRIPTION_SERVER_SECRET', "your_secret_token"),
        'HEALTH_CHECK_PORT': get_variable('TRANSCRIPTION_SERVER_HEALTH_CHECK_PORT', 8001, validate_int),
        'PROMETHEUS_PORT': get_variable('TRANSCRIPTION_SERVER_PROMETHEUS_PORT', 2112, validate_int),
    }

    if not valid_config:
        logging.error("Invalid config. Please fix the errors and try again.")
        sys.exit(1)

    return settings

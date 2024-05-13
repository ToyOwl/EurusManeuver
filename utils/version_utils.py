import platform
import logging
import pkg_resources as pkg
import torch

LOGGER = logging.getLogger('EUROS-MANEUVER')

def emojis(string=''):
    return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string

def check_version(current='0.0.0', minimum='0.0.0', name='version', pinned=False, hard=False, verbose=False):
    current_version, minimum_version = pkg.parse_version(current), pkg.parse_version(minimum)
    result = current_version == minimum_version if pinned else current_version >= minimum_version
    message = f'WARNING ⚠ {name} {minimum} is required by EurosManeuver, but {name} {current} is currently installed'
    if hard:
        assert result, emojis(message)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(message)
    return result

def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    def decorator(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)
    return decorator

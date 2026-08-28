"""
This script is a key part of the following publications:
    - Herig Coimbra, Pedro Henrique and Loubet, Benjamin and Laurent, Olivier and Mauder, Matthias and Heinesch, Bernard and 
    Bitton, Jonathan and Delpierre, Nicolas and Depuydt, Jérémie and Buysse, Pauline, Improvement of Co2 Flux Quality Through 
    Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. 
    Available at SSRN: https://ssrn.com/abstract=4642939 or http://dx.doi.org/10.2139/ssrn.4642939
"""
# standard modules
import os
import logging
import datetime


def start_logging(outputpath, *, level=logging.INFO, **kwargs):
    """Attach a file handler to the ``fluxprint`` logger (opt-in, for applications).

    A library must not reconfigure the root logger, so this attaches a handler to
    the package logger only and leaves root handlers and warning capture
    untouched. Intended to be called explicitly by an application or CLI, never
    at import time.

    Args:
        outputpath: Directory under which a timestamped ``log/`` file is created.
        level: Level to set on the ``fluxprint`` logger.
        **kwargs: Reserved for backwards compatibility (ignored).

    Returns:
        logging.Logger: The configured ``fluxprint`` logger.
    """
    logname = str(os.path.join(
        outputpath, f"log/current_{datetime.datetime.now().strftime('%y%m%dT%H%M%S')}.log"))
    mkdirs(logname)

    handler = logging.FileHandler(logname, mode="a")
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

    fp_logger = logging.getLogger("fluxprint")
    fp_logger.addHandler(handler)
    fp_logger.setLevel(level)
    fp_logger.info("Logging fluxprint to %s", logname)
    return fp_logger


def mkdirs(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def update_nested_dict(d, u):
    """
    Recursively updates a nested dictionary `d` with values from another dictionary `u`.
    If a key in `u` maps to a dictionary and the corresponding key in `d` also maps to a dictionary,
    the function updates the nested dictionary in `d`. Otherwise, it overwrites the value in `d`.

    Args:
        d (dict): The dictionary to update.
        u (dict): The dictionary containing updates.

    Returns:
        dict: The updated dictionary.
    """
    # Iterate over each key-value pair in the update dictionary `u`
    for k, v in u.items():
        # Check if the current value is a dictionary
        if isinstance(v, dict):
            # If the corresponding value in `d` is also a dictionary, recursively update it
            # Use `d.get(k, {})` to handle cases where the key `k` is not already in `d`
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            # If the value is not a dictionary, directly update/overwrite the key in `d`
            d[k] = v
    # Return the updated dictionary
    return d

"""
Exceptions

For original code see  Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
contact: natascha.kljun@cec.lu.se
"""
import logging
import numbers

import numpy as np

logger = logging.getLogger('fluxprint.exceptions')

__all__ = ['FluxPrintError', 'InputValidationError', 'exTypes', 'exceptions',
           'check_ffp_inputs', 'raise_ffp_exception']


class FluxPrintError(Exception):
    """Base class for all fluxprint errors."""


class InputValidationError(FluxPrintError, ValueError):
    """A model input failed validation.

    ``code`` is the numeric code from the FFP exception table, when applicable.
    """

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


exTypes = {'message': 'Message',
           'alert': 'Alert',
           'error': 'Error',
           'fatal': 'Fatal error'}

exceptions = [
    {'code': 1,
     'type': exTypes['fatal'],
     'msg': 'At least one required parameter is missing. Please enter all '
            'required inputs. Check documentation for details.'},
    {'code': 2,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be larger than zero.'},
    {'code': 3,
     'type': exTypes['error'],
     'msg': 'z0 (roughness length) must be larger than zero.'},
    {'code': 4,
     'type': exTypes['error'],
     'msg': 'h (BPL height) must be larger than 10 m.'},
    {'code': 5,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be smaller than h (PBL height).'},
    {'code': 6,
     'type': exTypes['alert'],
     'msg': 'zm (measurement height) should be above roughness sub-layer (12.5*z0).'},
    {'code': 7,
     'type': exTypes['error'],
     'msg': 'zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.'},
    {'code': 8,
     'type': exTypes['error'],
     'msg': 'sigmav (standard deviation of crosswind) must be larger than zero.'},
    {'code': 9,
     'type': exTypes['error'],
     'msg': 'ustar (friction velocity) must be >=0.1.'},
    {'code': 10,
     'type': exTypes['error'],
     'msg': 'wind_dir (wind direction) must be >=0 and <=360.'},
    {'code': 11,
     'type': exTypes['fatal'],
     'msg': 'Passed data arrays (ustar, zm, h, ol) don\'t all have the same length.'},
    {'code': 12,
     'type': exTypes['fatal'],
     'msg': 'No valid zm (measurement height above displacement height) passed.'},
    {'code': 13,
     'type': exTypes['alert'],
     'msg': 'Using z0, ignoring umean if passed.'},
    {'code': 14,
     'type': exTypes['alert'],
     'msg': 'No valid z0 passed, using umean.'},
    {'code': 15,
     'type': exTypes['fatal'],
     'msg': 'No valid z0 or umean array passed.'},
    {'code': 16,
     'type': exTypes['error'],
     'msg': 'At least one required input is invalid. Skipping current footprint.'},
    {'code': 17,
     'type': exTypes['alert'],
     'msg': 'Only one value of zm passed. Using it for all footprints.'},
    {'code': 18,
     'type': exTypes['fatal'],
     'msg': 'if provided, rs must be in the form of a number or a list of numbers.'},
    {'code': 19,
     'type': exTypes['alert'],
     'msg': 'rs value(s) larger than 90% were found and eliminated.'},
    {'code': 20,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be above roughness sub-layer (12.5*z0).'},
    {'code': 21,
     'type': exTypes['error'],
     'msg': 'Missing or non-finite input (None, NaN or inf). Skipping current footprint.'},
]


def _is_finite_number(value):
    return (isinstance(value, numbers.Number)
            and bool(np.isfinite(value)))


def check_ffp_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer,
                     verbosity, *, quiet=False):
    # Check passed values for physical plausibility and consistency.
    # quiet=True skips the per-failure reporting (raise_ffp_exception logs one
    # record per rejected input, which the mapped/dask path must not do per
    # record); the checks themselves are identical.
    report = (lambda code, verbosity: None) if quiet else raise_ffp_exception
    # Reject missing/non-finite records first: every comparison below is False
    # for NaN, so without this guard a NaN record would sail through validation
    # and silently poison the composited footprint.
    required = [ustar, sigmav, h, ol, wind_dir, zm,
                umean if z0 is None else z0]
    if not all(_is_finite_number(val) for val in required):
        report(21, verbosity)
        return False
    if zm <= 0.:
        report(2, verbosity)
        return False
    if z0 is not None and umean is None and z0 <= 0.:
        report(3, verbosity)
        return False
    if h <= 10.:
        report(4, verbosity)
        return False
    if zm > h :
        report(5, verbosity)
        return False
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        if rslayer == 1:
            report(6, verbosity)
        else:
            report(20, verbosity)
            return False
    # ol == 0 is unphysical and would raise ZeroDivisionError; treat it like the
    # too-unstable case (zm/ol <= -15.5) and reject the record.
    if ol == 0 or float(zm) / ol <= -15.5:
        report(7, verbosity)
        return False
    if sigmav <= 0:
        report(8, verbosity)
        return False
    if ustar <= 0.1:
        report(9, verbosity)
        return False
    if wind_dir > 360:
        report(10, verbosity)
        return False
    if wind_dir < 0:
        report(10, verbosity)
        return False
    return True


def raise_ffp_exception(code, verbosity=1):
    """Raise (fatal codes) or report (alert/error codes) an FFP exception.

    Fatal codes raise :class:`InputValidationError` carrying the full message
    regardless of ``verbosity`` -- verbosity gates console printing only, never
    the message content. Error codes are logged at warning level (visible once
    the application configures logging); routine alerts/messages at info
    level. Non-fatal codes are also printed when ``verbosity > 1``.
    """
    ex = [it for it in exceptions if it['code'] == code][0]
    string = ex['type'] + '(' + str(ex['code']).zfill(4) + '):\n ' + ex['msg']

    if ex['type'] == exTypes['fatal']:
        raise InputValidationError(
            string + '\n FFP_fixed_domain execution aborted.', code=code)

    string = string + '\n Execution continues.'
    if ex['type'] == exTypes['error']:
        logger.warning('%s', string)
    else:  # routine alerts/messages must not read like data problems
        logger.info('%s', string)
    if verbosity > 1:
        print(string)
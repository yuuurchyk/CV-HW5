import logging
import os
from types import SimpleNamespace


import dotenv
import jsonschema


def _get_env() -> SimpleNamespace:
    """Parses and validates entries in .env file

    Returns:
        namedtuple: validated variables from .env file
    """
    logging.debug('Loading info from .env file')

    env = dotenv.dotenv_values()

    logging.debug('Checking keys')

    schema = {
        'type': 'object',
        'properties': {
            'DATASETS_ROOT': {'type': 'string'},
            'EXPS_ROOT': {'type': 'string'}
        },
        'additionalProperties': False,
        'required': [
            'DATASETS_ROOT',
            'EXPS_ROOT'
        ]
    }

    try:
        jsonschema.validate(instance=env, schema=schema)
    except Exception:
        logging.error('Failed to validate .env schema')
        raise

    env = SimpleNamespace(**env)

    assert os.path.isdir(env.DATASETS_ROOT), '%s from .env file is not a folder' % (
        env.DATASETS_ROOT, )
    assert os.path.isdir(env.EXPS_ROOT), '%s from .env file is not a folder' % (
        env.EXPS_ROOT, )

    return env


def get_datasets_root() -> str:
    env = _get_env()
    return env.DATASETS_ROOT


def get_exps_root() -> str:
    env = _get_env()
    return env.EXPS_ROOT

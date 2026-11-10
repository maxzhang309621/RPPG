"""YAML-based configuration system for PhysNet."""

import yaml


class ConfigDict(dict):
    """A dictionary that allows attribute-style access."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, ConfigDict):
            val = ConfigDict(val)
            self[key] = val
        return val

    def __setattr__(self, key, val):
        self[key] = val


def _convert_to_config_dict(d):
    """Recursively convert nested dicts to ConfigDict."""
    if isinstance(d, dict):
        cd = ConfigDict()
        for k, v in d.items():
            cd[k] = _convert_to_config_dict(v)
        return cd
    elif isinstance(d, list):
        return [_convert_to_config_dict(item) for item in d]
    return d


def deep_merge(base, override):
    """Deep-merge override into base. Returns a new ConfigDict.

    For nested dicts, merges recursively. For other types, override wins.
    """
    result = ConfigDict()
    for k, v in base.items():
        result[k] = v
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return _convert_to_config_dict(result)


def config_to_dict(cfg):
    """Recursively convert ConfigDict to plain dict for YAML serialization."""
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [config_to_dict(item) for item in cfg]
    return cfg


def load_config(yaml_path):
    """Load a YAML config file and return a ConfigDict."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)
    return _convert_to_config_dict(raw)


def get_model_frame_num(config):
    """Extract the model's expected frame count from merged config.

    Returns:
        int or None: The model's frame number, or None if not determinable.
    """
    model_name = config.MODEL.NAME
    if model_name == "FactorizePhys":
        return config.MODEL.FACTORIZEPHYS.FRAME_NUM
    elif model_name == "EfficientPhys":
        return config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
    elif model_name == "Physnet":
        return config.MODEL.PHYSNET.FRAME_NUM
    return None


def sync_chunk_length(config):
    """Sync PREPROCESS.CHUNK_LENGTH with the model's frame number.

    Different models expect different chunk lengths (PhysNet=128,
    FactorizePhys/EfficientPhys=160). This sets CHUNK_LENGTH in all
    data splits to match the model, so the same dataset config works
    for any model.
    """
    frame_num = get_model_frame_num(config)
    if frame_num is None:
        return

    for split_name in ('TRAIN', 'VALID', 'TEST'):
        split = getattr(config, split_name, None)
        if split is None:
            continue
        data = getattr(split, 'DATA', None)
        if data is None:
            continue
        preprocess = getattr(data, 'PREPROCESS', None)
        if preprocess is not None and hasattr(preprocess, 'CHUNK_LENGTH'):
            preprocess.CHUNK_LENGTH = frame_num

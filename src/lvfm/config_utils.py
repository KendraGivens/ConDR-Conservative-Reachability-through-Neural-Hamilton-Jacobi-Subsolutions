from types import SimpleNamespace


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(v) for v in value]
    return value


def cfg_get(cfg, path, default=None):
    cur = cfg
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def cfg_has(cfg, path):
    sentinel = object()
    return cfg_get(cfg, path, sentinel) is not sentinel


def as_dict(value):
    if isinstance(value, SimpleNamespace):
        return {k: as_dict(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: as_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_dict(v) for v in value]
    return value

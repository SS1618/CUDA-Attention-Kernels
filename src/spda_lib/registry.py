SPDA_REGISTRY = {}

def register_variants(name):
    def decorator(func):
        SPDA_REGISTRY[name] = func
        return func
    return decorator

def execute_spda_variant(name, *args, **kwargs):
    if name not in SPDA_REGISTRY:
        raise ValueError(f"SPDA variant '{name}' is not registered.")
    return SPDA_REGISTRY[name](*args, **kwargs)
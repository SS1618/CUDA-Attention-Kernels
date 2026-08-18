SDPA_REGISTRY = {}

def register_variants(name):
    def decorator(func):
        SDPA_REGISTRY[name] = func
        return func
    return decorator

def create_sdpa_variant(name):
    if name not in SDPA_REGISTRY:
        raise ValueError(f"SDPA variant '{name}' is not registered.")
    return SDPA_REGISTRY[name]().forward  # Return the forward method of the instance
def execute_sdpa_variant(name, *args, **kwargs):
    if name not in SDPA_REGISTRY:
        raise ValueError(f"SDPA variant '{name}' is not registered.")
    return SDPA_REGISTRY[name](*args, **kwargs)
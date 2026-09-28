import os
from supabase import create_client, Client

_module_client = None

def get_supabase() -> Client:
    """Create a fresh Supabase client to avoid stale HTTP connections."""
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

def __getattr__(name):
    """
    Lazy module-level client — only created when first accessed, not at
    import time. This prevents a crash if env vars aren't set yet during
    cold start on Render.
    """
    if name == "supabase":
        global _module_client
        if _module_client is None:
            _module_client = get_supabase()
        return _module_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

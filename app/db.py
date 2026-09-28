import os
from supabase import create_client, Client

def get_supabase() -> Client:
    """Create a fresh Supabase client to avoid stale HTTP connections."""
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

# Keep module-level reference for backwards compat (used by services)
supabase = get_supabase()

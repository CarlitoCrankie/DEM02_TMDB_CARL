import importlib.util
import os

# Load new_config dynamically
config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'new_config.py')
)

spec = importlib.util.spec_from_file_location("new_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

TMDB_API_KEY = config.TMDB_API_KEY

print(TMDB_API_KEY)
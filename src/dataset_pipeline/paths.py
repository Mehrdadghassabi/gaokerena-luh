"""
Where the pipeline reads and writes. Everything is relative to this folder, so
the folder can be moved or copied anywhere.

Override with environment variables:
    LUH_DATA_DIR  input/output xlsx files    (default: <this folder>/data)
    LUH_WORK_DIR  stage-2 runs + API caches  (default: <this folder>/workdir)

API keys come from the environment (which <this folder>/.env is loaded into,
without overriding variables already set), falling back to a key file here:
    OPENROUTER_API_KEY  or  OPENROUTER_KEY.txt
    DEEPSEEK_API_KEY    or  DEEPSEEK_KEY.txt
"""
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path):
    """Minimal KEY=VALUE reader, so .env needs no extra dependency."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            os.environ.setdefault(key, value.strip().strip("'\""))


load_dotenv(os.path.join(PIPELINE_DIR, ".env"))

DATA_DIR = os.environ.get("LUH_DATA_DIR") or os.path.join(PIPELINE_DIR, "data")
WORK_DIR = os.environ.get("LUH_WORK_DIR") or os.path.join(PIPELINE_DIR, "workdir")


def read_api_key(env_var, key_file):
    key = os.environ.get(env_var)
    if key:
        return key.strip()
    path = os.path.join(PIPELINE_DIR, key_file)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    raise SystemExit(f"No API key: set {env_var}, or put the key in {path}")

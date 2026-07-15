import json
import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_lista():
    # Get variables from .env
    api_url = os.getenv('SUPABASE_URL')
    api_key = os.getenv('api_key')
    authorization = os.getenv('authorization')

    # Build a curl command using the Supabase URL and auth headers
    command = [
        'curl',
        '-s',
        '-H', f'apikey: {api_key}',
        '-H', f'Authorization: Bearer {authorization}',
        f'{api_url}/rest/v1/trajectory'
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

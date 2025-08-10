
import os
from dotenv import load_dotenv

# Load .env file into environment variables
load_dotenv()

# Access variables from environment
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

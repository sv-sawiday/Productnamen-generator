from dotenv import load_dotenv
import os

# Load .env file contents into environment viraibles
load_dotenv()

# Read specific environment variables
API_KEY = os.getenv("API_KEY")
DATA_PATH = os.getenv("DATA_PATH")

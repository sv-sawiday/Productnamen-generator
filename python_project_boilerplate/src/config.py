from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
API_KEY = os.getenv("API_KEY")
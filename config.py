import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
PRIMARY_API_KEY = os.getenv('PRIMARY_API_KEY')
SECONDARY_API_KEY = os.getenv('SECONDARY_API_KEY')
BASE_URL = os.getenv('BASE_URL', 'https://chat-ai.academiccloud.de/v1') 


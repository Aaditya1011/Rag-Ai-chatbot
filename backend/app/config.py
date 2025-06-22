from dotenv import load_dotenv
import os
load_dotenv()

#hf_token = os.getenv('HF_TOKEN')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
BACKEND_API_URL= os.getenv('BACKEND_API_URL','http://127.0.0.1:8000')
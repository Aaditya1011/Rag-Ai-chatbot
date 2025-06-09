from dotenv import load_dotenv
import os
load_dotenv()

hf_token = os.getenv('HF_TOKEN')
together_api_key = os.getenv('TOGETHER_API_KEY')

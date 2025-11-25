import asyncio
import os
import uuid

from audio_pre_processing import pre_process_call
from helper import S3Service
from helper import get_logger
from spech_to_text import SarvamSpeechService
from dotenv import load_dotenv
import tempfile
import json

load_dotenv()

logger = get_logger(os.path.basename(__file__))


_s3service = S3Service()
_sarvamclinet = SarvamSpeechService(os.getenv('API_SUBSCRIPTION_KEY'),
                                    output_dir=tempfile.gettempdir())

def get_file_from_s3(bucket: str, s3_key: str) -> str:
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key))
    _s3service.download_file(bucket, s3_key, tmp)
    return tmp

def get_transcript_from_stt(path: str):
    file_paths = asyncio.run(_sarvamclinet.transcribe([path], "hi-IN"))
    return file_paths

def uploas_transcript(bucket: str, s3_key: str, filepath :str) -> str:
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
    except Exception as e:
        logger.error(f"File {filepath} is not a valid JSON file. Error: {e}")
        return None
    _s3service.upload_file(bucket, s3_key, filepath)
    return filepath

def uploas_pre_processed_audio(bucket: str, s3_key: str, filepath :str) -> str:
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist.")
        return None
    _s3service.upload_file(bucket, s3_key, filepath)
    return filepath

if __name__=="__main__":
    x = get_file_from_s3(os.getenv('S3_BUCKET_NAME') , 'VoiceAI/recording/4_nov_2025_1763414527466.mp3')
    request_id = str(uuid.uuid4())
    processed_path = pre_process_call(x, request_id)
    path = uploas_pre_processed_audio(os.getenv('S3_BUCKET_NAME'),
                                      f"VoiceAI/processed/{x.split('/')[-1].split('.')[0]}.mp3", processed_path)
    files = get_transcript_from_stt(processed_path)
    result_key = [uploas_transcript(
        os.getenv('S3_BUCKET_NAME'),
        f"VoiceAI/transcript/{f.split('/')[-1].split('.')[0]}.json",
        os.path.join(_sarvamclinet.output_dir, f)) for f in files]
    logger.info(f"{result_key}")

from helper import S3Service
from helper import get_logger
from spech_to_text import SarvamSpeechService
from dotenv import load_dotenv
import tempfile

load_dotenv()

logger = get_logger(__name__)


_s3service = S3Service()

def get_file_from_s3(bucket: str, s3_key: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    logger.info(f"{tmp}")
    _s3service.download_file(bucket, s3_key, tmp)
    return tmp

if __name__=="__main__":
    x = get_file_from_s3('weye-stage-2', 'VoiceAI/recording/20251030163646.mp3')
    logger.info(f"{x}")

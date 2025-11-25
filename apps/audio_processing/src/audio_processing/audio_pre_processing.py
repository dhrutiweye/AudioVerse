import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from helper import get_logger, S3Service
from dotenv import load_dotenv
from database import get_documents_by_filter, insert_documents, update_document
from audio_processor import AudioProcessor

load_dotenv()

logger = get_logger(os.path.basename(__file__))
# _s3service = S3Service()


# def get_pending_calls_for_preprocessing() -> List[Dict[str, Any]]:
#     """Get pending calls for preprocessing using atomic find-and-modify operation"""
#     try:
#         # Calculate timestamp for 30 minutes ago
#         thirty_mins_ago = datetime.utcnow() - timedelta(minutes=30)
#
#         # Use findAndModify (find_one_and_update) to atomically find and lock ONE call
#         # This ensures each pod gets a unique call even if they run simultaneously
#         updated_call = await unprocessed_calls.find_one_and_update(
#             {
#                 "status": "pending",
#                 "created_at": {"$gte": thirty_mins_ago}
#             },
#             {
#                 "$set": {
#                     "status": "preprocessing",
#                     "updated_at": datetime.utcnow(),
#                     "pod_id": os.environ.get("HOSTNAME", "unknown_pod"),
#                     "lock_acquired_at": datetime.utcnow()
#                 }
#             },
#             sort=[("created_at", 1)],  # Get oldest first
#             return_document=ReturnDocument.AFTER  # Return the updated document
#         )
#
#         if not updated_call:
#             logger.info("No pending calls found for preprocessing")
#             return []
#
#         call_id = updated_call["_id"]
#         logger.info(f"[POD: {os.environ.get('HOSTNAME', 'unknown')}] Locked call {call_id} for preprocessing")
#         return [updated_call]
#
#     except Exception as e:
#         logger.error(
#             f"[POD: {os.environ.get('HOSTNAME', 'unknown')}] Error getting pending call for preprocessing: {str(e)}")
#         return []


def pre_process_call(raw_path, request_id) -> Optional[str]:
    audio_processor = AudioProcessor()
    success, processed_path, error = asyncio.run(audio_processor.process_audio(
        raw_path,
        request_id,
        sample_rate=22050,
        mp3_bitrate='8k'
    ))

    if not success or not processed_path:
        raise Exception(f"Audio processing failed: {error}")

    logger.info(f"Successfully preprocessed call {request_id}")
    return processed_path


async def preprocess_call(call: Dict[str, Any]) -> None:
    """Preprocess a single call record"""
    call_id = str(call["_id"])
    logger.info(f"\nPreprocessing call {call_id}")
    pass

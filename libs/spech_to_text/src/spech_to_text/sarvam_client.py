import os
from sarvamai import AsyncSarvamAI
from helper.logger import get_logger

class SarvamSpeechService:
    def __init__(
        self,
        api_key: str,
        language_code: str = "en-IN",
        model: str = "saarika:v2.5",
        with_diarization: bool = True,
        num_speakers: int = 2,
        output_dir: str = "./output"
    ):
        """
        Initializes the Sarvam Speech-to-Text service once.
        """
        self.client = AsyncSarvamAI(api_subscription_key=api_key)
        self.language_code = language_code
        self.model = model
        self.with_diarization = with_diarization
        self.num_speakers = num_speakers
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__)

        os.makedirs(output_dir, exist_ok=True)

    async def transcribe(self, audio_paths: list[str], language_code: str):
        """
        Runs a single transcription job for one audio file.
        """

        self.logger.info(f"📌 Starting STT job for: {audio_paths}")

        job = await self.client.speech_to_text_job.create_job(
            language_code=language_code or self.language_code,
            model=self.model,
            with_diarization=self.with_diarization,
            num_speakers=self.num_speakers
        )

        # Upload audio
        await job.upload_files(audio_paths)

        # Start processing
        await job.start()
        self.logger.info(f"Starting STT {audio_paths} with job id {job.job_id}")
        await job.wait_until_complete()

        if await job.is_failed():
            self.logger.error(f"❌ STT job failed for: {audio_paths}")
            return None

        # Download final output
        await job.download_outputs(output_dir=self.output_dir)

        self.logger.info(f"✅ Output saved to {self.output_dir}")
        return True

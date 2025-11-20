import asyncio

from sarvam_client import SarvamSpeechService


async def main():
    service = SarvamSpeechService(
        api_key="sk_vb3ohrdq_fGv55kuexWshakBpNStR8jsD",
        output_dir="./output"
    )

    audio_files = [
        "/Users/weye/Documents/code/python/helper/20251030163646.mp3",
        # "/Users/weye/Documents/code/python/helper/20251027160031.mp3"
    ]

    # for audio in audio_files:
    await service.transcribe(audio_files, "hi-IN")

if __name__ == "__main__":
    asyncio.run(main())

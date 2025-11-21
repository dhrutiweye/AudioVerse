import asyncio

from sarvam_client import SarvamSpeechService


async def main():
    service = SarvamSpeechService(
        api_key="${your_key}",
        output_dir="./output"
    )

    audio_files = [
        "/Users/weye/Documents/code/python/helper/20251030163646.mp3",
        # "/Users/weye/Documents/code/python/helper/20251027160031.mp3"
    ]

    # for audio in audio_files:
    return await service.transcribe(audio_files, "hi-IN")


if __name__ == "__main__":
    # print(asyncio.run(main()))
    pass

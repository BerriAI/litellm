# What is this?
## Tests `litellm.transcription` endpoint
import pytest
import asyncio, time
import aiohttp
from openai import AsyncOpenAI
import sys, os, dotenv
from typing import Optional
from dotenv import load_dotenv

audio_file = open("./gettysburg.wav", "rb")

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm


def test_transcription():
    transcript = litellm.transcription(model="whisper-1", file=audio_file)
    print(f"transcript: {transcript}")


test_transcription()

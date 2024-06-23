"""
This script implements a real-time audio-visual assistant using screen capture and microphone input.
It captures screenshots, listens for audio input, processes the input using either GPT-4o, Claude 3.5 Sonnet
or Gemini 1.5 Flash, and provides spoken responses.

The assistant is capable of understanding spoken German input and responding in German,
while also considering visual context from the screen capture.

Usage:
    python script_name.py --model [gpt4o|claude|gemini] --language [english|german|chinese]
Dependencies:
- mss for screen capture
- PyAudio for audio output
- SpeechRecognition for audio input processing
- LangChain for AI model integration
- OpenAI's GPT-4o, Anthropic's Claude 3.5 Sonnet, or Google's Gemini 1.5 Flash for natural language processing
- OpenAI's Whisper for speech recognition
- OpenAI's TTS for text-to-speech conversion

Make sure to set up the necessary API keys in your environment variables or .env file.
"""

import argparse
import base64
from io import BytesIO
import time

import openai
from PIL import Image
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from mss import mss
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Load environment variables from .env file
load_dotenv()

class ScreenCapture:
    """
    A class to handle screen capturing.
    """

    def __init__(self):
        """Initialize the ScreenCapture with mss instance."""
        pass

    def capture(self, encode=False):
        """
        Capture the entire screen.

        Args:
            encode (bool): If True, encode the image as base64 JPEG.

        Returns:
            PIL.Image or bytes: The captured screen image or its base64 encoded version.
        """
        with mss() as sct:
            screenshot = sct.grab(sct.monitors[0])
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            if encode:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue())

            return img

class Assistant:
    """
    A class representing an AI assistant capable of processing audio-visual input
    and providing spoken responses.
    """

    def __init__(self, model, language):
        """
        Initialize the Assistant with a specified language model.

        Args:
            model: An instance of a language model compatible with LangChain.
        """
        self.language = language
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """
        Process the user's prompt and screen capture to generate a response.

        Args:
            prompt (str): The user's spoken input, transcribed to text.
            image (bytes): Base64 encoded image from the screen capture.
        """
        if not prompt:
            return

        print("Prompt:", prompt)

        # Generate a response using the language model
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        """
        Convert the text response to speech and play it.

        Args:
            response (str): The text to be converted to speech.
        """
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        """
        Create a LangChain inference chain for processing inputs and generating responses.

        Args:
            model: An instance of a language model compatible with LangChain.

        Returns:
            RunnableWithMessageHistory: A LangChain runnable chain with message history.
        """
        SYSTEM_PROMPT = f"""
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. The image is a screenshot
        of the user's entire screen.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.
        
        If the answer would be too long, summarise it.
        
        Always answer in {self.language}.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AI Assistant with screen capture and microphone input.")
    parser.add_argument('--model', type=str, choices=['gpt4o', 'claude', 'gemini'], default='claude',
                        help="Specify the model to use: 'gpt4o' for GPT-4o, 'claude' for Claude 3.5 Sonnet, or 'gemini' for Gemini 1.5 Flash")
    parser.add_argument('--language', type=str, choices=['english', 'german', 'chinese'], default='english',
                        help="Specify the language for input and output: 'english', 'german', or 'chinese'")
    return parser.parse_args()

def initialize_model(model_choice):
    """
    Initialize and return the specified language model.

    Args:
        model_choice (str): The model choice ('gpt4o', 'claude', or 'gemini')

    Returns:
        LangChain compatible model instance
    """
    if model_choice == 'gpt4o':
        return ChatOpenAI(model="gpt-4o")
    elif model_choice == 'claude':
        return ChatAnthropic(model="claude-3-sonnet-20240229")
    else:  # 'gemini'
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def main():
    """
    Main function to run the AI assistant.
    """
    args = parse_arguments()

    # Initialize the screen capture
    screen_capture = ScreenCapture()

    # Initialize the chosen model
    model = initialize_model(args.model)
    print(f"Using model: {args.model}")
    print(f"Using language: {args.language}")

    # Create an instance of the Assistant
    assistant = Assistant(model, args.language)

    def audio_callback(recognizer, audio):
        """
        Callback function to process audio input.

        Args:
            recognizer (speech_recognition.Recognizer): The speech recognizer object.
            audio (speech_recognition.AudioData): The captured audio data.
        """
        try:
            # Transcribe the audio using Whisper
            prompt = recognizer.recognize_whisper(audio, model="base", language=args.language)
            # Process the transcribed text and screen capture
            assistant.answer(prompt, screen_capture.capture(encode=True))

        except UnknownValueError:
            print("There was an error processing the audio.")

    # Set up the speech recognizer and microphone
    recognizer = Recognizer()
    microphone = Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    # Start listening for audio input in the background
    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    print(f"AI Assistant is running. Speak in {args.language} to interact. Press Ctrl+C to exit.")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping the AI Assistant...")

    # Clean up resources
    stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()
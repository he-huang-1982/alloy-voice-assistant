"""
This script implements a real-time audio-visual assistant using a webcam and microphone.
It captures video, listens for audio input, processes the input using Claude 3.5 Sonnet,
and provides spoken responses.

The assistant is capable of understanding spoken German input and responding in German,
while also considering visual context from the webcam feed.

Dependencies:
- OpenCV (cv2) for webcam handling
- PyAudio for audio output
- SpeechRecognition for audio input processing
- LangChain for AI model integration
- Anthropic's Claude 3.5 Sonnet for natural language processing
- OpenAI's Whisper for speech recognition
- OpenAI's TTS for text-to-speech conversion

Make sure to set up the necessary API keys in your environment variables or .env file.
"""

import base64
from threading import Lock, Thread

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_anthropic import ChatAnthropic
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Load environment variables from .env file
load_dotenv()


class WebcamStream:
    """
    A class to handle webcam streaming in a separate thread.
    This allows for non-blocking webcam reads.
    """

    def __init__(self):
        """Initialize the WebcamStream with default settings."""
        self.stream = VideoCapture(index=0)  # Open default camera
        _, self.frame = self.stream.read()  # Read first frame
        self.running = False
        self.lock = Lock()  # For thread-safe operations

    def start(self):
        """Start the webcam stream in a separate thread."""
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        """Continuously update the frame from the webcam."""
        while self.running:
            _, frame = self.stream.read()
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        """
        Read the current frame from the webcam.

        Args:
            encode (bool): If True, encode the frame as base64 JPEG.

        Returns:
            numpy.ndarray or bytes: The current frame or its base64 encoded version.
        """
        with self.lock:
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        """Stop the webcam stream and wait for the thread to finish."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Release the webcam when the object is destroyed."""
        self.stream.release()


class Assistant:
    """
    A class representing an AI assistant capable of processing audio-visual input
    and providing spoken responses.
    """

    def __init__(self, model):
        """
        Initialize the Assistant with a specified language model.

        Args:
            model: An instance of a language model compatible with LangChain.
        """
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """
        Process the user's prompt and webcam image to generate a response.

        Args:
            prompt (str): The user's spoken input, transcribed to text.
            image (bytes): Base64 encoded image from the webcam.
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
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.

        Always answer in German.
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


# Initialize the webcam stream
webcam_stream = WebcamStream().start()

# Initialize the Claude 3.5 Sonnet model
model = ChatAnthropic(model="claude-3-sonnet-20240229")

# Create an instance of the Assistant
assistant = Assistant(model)


def audio_callback(recognizer, audio):
    """
    Callback function to process audio input.

    Args:
        recognizer (speech_recognition.Recognizer): The speech recognizer object.
        audio (speech_recognition.AudioData): The captured audio data.
    """
    try:
        # Transcribe the audio using Whisper
        prompt = recognizer.recognize_whisper(audio, model="base", language="german")
        # Process the transcribed text and webcam image
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


# Set up the speech recognizer and microphone
recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

# Start listening for audio input in the background
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# Main loop to display webcam feed and handle user input
while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:  # Exit on 'Esc' or 'q' key press
        break

# Clean up resources
webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
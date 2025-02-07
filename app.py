import os
import tempfile
import logging
from logging import StreamHandler
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yt_dlp
import shutil
import whisper
import time
from typing import Optional, Tuple, List, Dict
import torch
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create loggers 
logger = logging.getLogger(__name__)
ytdl_logger = logging.getLogger('yt-dlp')
ytdl_logger.addHandler(StreamHandler())
ytdl_logger.setLevel(logging.INFO)

class YTDLLogger:
    def debug(self, msg):
        if msg.startswith('[debug] '):
            ytdl_logger.debug(msg)
        else:
            ytdl_logger.info(msg)
    
    def info(self, msg):
        ytdl_logger.info(msg)
    
    def warning(self, msg):
        ytdl_logger.warning(msg)
    
    def error(self, msg):
        ytdl_logger.error(msg)

class IdiomProcessor:
    def __init__(self, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.model = AutoModel.from_pretrained("xlm-roberta-base")
        
    def get_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
    def detect_idioms(self, text: str) -> bool:
        words = text.split()
        if len(words) < 2:
            return False
            
        word_embeddings = [self.get_embeddings(word) for word in words]
        phrase_embedding = self.get_embeddings(text)
        
        word_centroid = torch.stack(word_embeddings).mean(dim=0)
        similarity = torch.cosine_similarity(word_centroid, phrase_embedding)
        
        return similarity.item() < 0.7
    
    def improve_translation(self, text: str) -> str:
        if self.detect_idioms(text):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, num_beams=5, max_length=50)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

app = Flask(__name__)

def extract_video_id(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def is_valid_youtube_url(url):
    """Validate YouTube URL format"""
    return bool(extract_video_id(url))

# Ensure the temp directory exists
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Load whisper model
model = whisper.load_model("base")

def progress_hook(d):
    if d['status'] == 'downloading':
        try:
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes', 0)
            total_str = f"{total / 1024 / 1024:.1f}MB" if total else "N/A"
            downloaded_str = f"{downloaded / 1024 / 1024:.1f}MB"
            eta = d.get('_eta_str', 'N/A')
            logger.info(
                f"Download progress: {percent} ({downloaded_str}/{total_str}) "
                f"Speed: {speed}, ETA: {eta}"
            )
        except Exception as e:
            logger.warning(f"Error logging progress: {e}")
    elif d['status'] == 'finished':
        logger.info(f"Download completed: {d.get('filename', 'unknown file')}")
    elif d['status'] == 'error':
        logger.error(f"Download error: {d.get('error', 'Unknown error')}")

def get_youtube_audio(url: str, output_dir: str) -> Tuple[str, str]:
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        logger.error("FFmpeg not found in system PATH")
        raise ValueError("FFmpeg is required but not found")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'logger': YTDLLogger(),
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'progress_hooks': [progress_hook],
        'keepvideo': False,
        'writethumbnail': False,
        'verbose': True,
        'retries': 5,
        'fragment_retries': 5,
        'ignoreerrors': False,
        'postprocessor_args': [
            '-acodec', 'libmp3lame',
            '-ar', '44100',
            '-ac', '2',
            '-b:a', '128k',
        ],
        'prefer_ffmpeg': True,
        'extract_flat': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', f"Video {info.get('id', 'unknown')}")
            
            ydl.params['http_headers'] = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            ydl.download([url])
            
            files = os.listdir(output_dir)
            mp3_files = [f for f in files if f.endswith('.mp3')]
            if not mp3_files:
                raise ValueError("No MP3 file was created")
            audio_file = os.path.join(output_dir, mp3_files[0])
            return audio_file, video_title
            
    except Exception as e:
        logger.error(f"Error in get_youtube_audio: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to process video: {str(e)}")

def process_whisper_output(segments: List[Dict], model_path: str = None) -> List[Dict]:
    processor = IdiomProcessor(model_path)
    return [
        {
            "text": processor.improve_translation(segment["text"].strip()),
            "start": segment["start"],
            "end": segment["end"]
        }
        for segment in segments
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-subtitles', methods=['POST'])
def generate_subtitles():
    temp_dir = None
    try:
        url = request.form.get('url')
        logger.info(f"Received request to generate subtitles for URL: {url}")
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
            
        if not is_valid_youtube_url(url):
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        audio_file, video_title = get_youtube_audio(url, temp_dir)

        result = model.transcribe(audio_file, task="translate")
        segments = result["segments"]
        
        # Process with improved translation
        formatted_subtitles = process_whisper_output(
            segments,
            model_path=os.getenv('TRANSLATION_MODEL_PATH')
        )

        video_id = extract_video_id(url)
        response_data = {
            'success': True,
            'subtitles': formatted_subtitles,
            'video_title': video_title,
            'video_id': video_id
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in generate_subtitles: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.route('/test-download', methods=['POST'])
def test_download():
    try:
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not is_valid_youtube_url(url):
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        try:
            audio_file, video_title = get_youtube_audio(url, temp_dir)
            file_size = os.path.getsize(audio_file)
            return jsonify({
                'success': True,
                'message': 'Download successful',
                'file_size': f"{file_size/1024/1024:.1f}MB",
                'video_title': video_title
            })
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        logger.error(f"Test download failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

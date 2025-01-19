import os
import tempfile
import logging
from logging import StreamHandler
import re
from flask import Flask, render_template, request, jsonify
import yt_dlp
import shutil
import whisper
import time
from typing import Optional, Tuple

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

app = Flask(__name__)

def extract_video_id(url):
    """Extract video ID from YouTube URL (including Shorts)"""
    patterns = [
        # Standard YouTube URLs
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        # YouTube Shorts URLs
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
    """Enhanced progress hook with detailed logging"""
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
            fragment_info = d.get('fragment_index', 0), d.get('fragment_count', 0)
            if all(fragment_info):
                logger.debug(f"Fragment progress: {fragment_info[0]}/{fragment_info[1]}")
        except Exception as e:
            logger.warning(f"Error logging progress: {e}")
    elif d['status'] == 'finished':
        logger.info(f"Download completed: {d.get('filename', 'unknown file')}")
    elif d['status'] == 'error':
        error_msg = d.get('error', 'Unknown error')
        logger.error(f"Download error: {error_msg}")

def get_youtube_audio(url: str, output_dir: str) -> Tuple[str, str]:
    """
    Download YouTube audio using yt-dlp
    Returns: Tuple of (audio_file_path, video_title)
    """
    # Verify FFmpeg installation
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        logger.error("FFmpeg not found in system PATH")
        raise ValueError("FFmpeg is required but not found")
    logger.info(f"Found FFmpeg at: {ffmpeg_path}")
    ydl_opts = {
        'format': 'bestaudio/best',  # Simplified format selection
        'logger': YTDLLogger(),
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',  # Lower quality for better reliability
        }],
        'progress_hooks': [progress_hook],
        'ffmpeg_location': None,  # Let yt-dlp find FFmpeg automatically
        'keepvideo': False,
        'writethumbnail': False,
        'verbose': True,
        'retries': 5,
        'fragment_retries': 5,
        'ignoreerrors': False,
        'postprocessor_args': [
            '-acodec', 'libmp3lame',  # Explicitly use MP3 codec
            '-ar', '44100',  # Standard sample rate
            '-ac', '2',      # Stereo
            '-b:a', '128k',  # Fixed bitrate
        ],
        'prefer_ffmpeg': True,
        'extract_flat': False,  # Disable playlist handling
    }
    
    try:
        logger.info(f"Starting download process for URL: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video information
            logger.info("Extracting video information...")
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', f"Video {info.get('id', 'unknown')}")
            logger.info(f"Video title: {video_title}")
            
            # Set user agent for download
            ydl.params['http_headers'] = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Download the audio
            logger.info("Initiating audio download...")
            ydl.download([url])
            
            # Get the output filename
            # Get the actual filename that yt-dlp created
            files = os.listdir(output_dir)
            mp3_files = [f for f in files if f.endswith('.mp3')]
            if not mp3_files:
                raise ValueError("No MP3 file was created")
            audio_file = os.path.join(output_dir, mp3_files[0])
            return audio_file, video_title
            
    except yt_dlp.utils.PostProcessingError as e:
        logger.error(f"FFmpeg post-processing error: {str(e)}", exc_info=True)
        raise ValueError(f"Audio conversion failed: {str(e)}")
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_youtube_audio: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to process video: {str(e)}")

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
            logger.warning("No URL provided in request")
            return jsonify({'error': 'No URL provided'}), 400
            
        if not is_valid_youtube_url(url):
            logger.warning(f"Invalid YouTube URL format: {url}")
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        # Create a temporary directory for this request
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        logger.debug(f"Created temporary directory: {temp_dir}")

        # Download YouTube video with retry logic
        logger.info("Initializing YouTube download")
        audio_file, video_title = get_youtube_audio(url, temp_dir)
        logger.debug(f"Audio downloaded to: {audio_file}")

        # Generate subtitles using whisper
        logger.info("Generating subtitles with Whisper")
        result = model.transcribe(audio_file, task="translate")
        # Get segments with timestamps
        segments = result["segments"]
        formatted_subtitles = [
            {
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"]
            }
            for segment in segments
        ]
        logger.debug("Subtitles generated successfully")

        video_id = extract_video_id(url)
        response_data = {
            'success': True,
            'subtitles': formatted_subtitles,
            'video_title': video_title,
            'video_id': video_id
        }
        logger.info("Successfully processed video")
        return jsonify(response_data)

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube download error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Failed to download video: {str(e)}"}), 500
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")
@app.route('/test-download', methods=['POST'])
def test_download():
    """Test route to verify YouTube download functionality"""
    try:
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not is_valid_youtube_url(url):
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        logger.info(f"Testing download for URL: {url}")
        
        try:
            audio_file, video_title = get_youtube_audio(url, temp_dir)
            file_size = os.path.getsize(audio_file)
            logger.info(f"Download successful. File size: {file_size/1024/1024:.1f}MB")
            
            return jsonify({
                'success': True,
                'message': 'Download successful',
                'file_size': f"{file_size/1024/1024:.1f}MB",
                'video_title': video_title
            })
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        logger.error(f"Test download failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

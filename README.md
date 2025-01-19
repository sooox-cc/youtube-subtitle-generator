# YouTube Subtitle Generator

A web application that generates and translates subtitles for YouTube videos. It can automatically translate subtitles from any language (including Korean) to English, and provides a synchronized viewing experience with the original video.

## Features

- 🎥 YouTube video playback with synchronized subtitles
- 🌐 Automatic translation to English from any language
- ⚡ Real-time subtitle highlighting and auto-scrolling
- 🎯 Support for regular YouTube videos and Shorts
- 📱 Responsive design for various screen sizes

## Prerequisites

Before installation, ensure you have the following:

- Python 3.11 specifically (required for OpenAI Whisper compatibility)
  - Other versions may cause dependency conflicts
  - Python 3.13 is not supported by required dependencies
- FFmpeg installed on your system
  - For Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - For Arch Linux: `sudo pacman -S ffmpeg`
  - For macOS: `brew install ffmpeg`
  - For Windows: Download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sooox-cc/youtube-subtitle-generator.git
   cd youtube-subtitle-generator
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment with Python 3.11 specifically
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter a YouTube URL (supports both regular videos and Shorts)

4. Click "Generate Subtitles" and wait for processing

5. Once complete, you'll see:
   - The YouTube video player at the top
   - Synchronized subtitles below the video
   - Auto-scrolling subtitles that follow the video playback

## How It Works

The application uses:

- **yt-dlp**: For downloading YouTube video audio
- **OpenAI's Whisper**: For speech recognition and translation
- **Flask**: For the web server
- **YouTube IFrame API**: For video playback

The process:
1. Downloads the audio from the YouTube video
2. Processes the audio through Whisper for transcription and translation
3. Generates timestamped subtitles
4. Displays the video with synchronized subtitles

## Notes

- Processing time depends on:
  - Video length
  - Your internet connection speed
  - Your computer's processing power
- The application runs locally and processes videos on your machine
- All temporary files are automatically cleaned up after processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

"""
RunPod Serverless Handler for Soccer Video Analysis
"""
import os
import json
import sys
import traceback
from pathlib import Path
import subprocess

# Add soccer examples to path
handler_dir = Path(__file__).parent
if (handler_dir / 'examples' / 'soccer').exists():
    sys.path.insert(0, str(handler_dir / 'examples' / 'soccer'))
else:
    # Already in soccer directory
    sys.path.insert(0, str(handler_dir))

# Import will happen after path setup
try:
    from tools.api_client import send_pass_data_with_summary
except ImportError:
    # Fallback if api_client not available
    def send_pass_data_with_summary(*args, **kwargs):
        return {"success": False, "error": "API client not available"}


def download_video(url: str, output_path: str) -> str:
    """Download video from URL"""
    try:
        import gdown
        if 'drive.google.com' in url:
            gdown.download(url, output_path, fuzzy=True)
        else:
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download video: {e}")


def ensure_models():
    """Ensure model files exist, download if needed"""
    # Determine correct paths based on where handler.py is located
    handler_dir = Path(__file__).parent
    
    # Check if we're in examples/soccer directory or at root
    if (handler_dir / 'examples' / 'soccer').exists():
        soccer_dir = handler_dir / 'examples' / 'soccer'
    elif (handler_dir / 'data').exists():
        soccer_dir = handler_dir
    else:
        # Try to find it
        soccer_dir = Path('/app/examples/soccer') if Path('/app').exists() else handler_dir
    
    data_dir = soccer_dir / 'data'
    models = [
        'football-player-detection.pt',
        'football-ball-detection.pt',
        'football-pitch-detection.pt'
    ]
    
    missing = [m for m in models if not (data_dir / m).exists()]
    if missing:
        print(f"Downloading missing models: {missing}")
        setup_script = soccer_dir / 'setup.sh'
        if setup_script.exists():
            subprocess.run(['bash', str(setup_script)], check=True, cwd=str(soccer_dir))
        else:
            print(f"⚠️  setup.sh not found at {setup_script}, models must be downloaded manually")


def handler(event):
    """
    RunPod serverless handler
    
    Expected input format:
    {
        "input": {
            "video_url": "https://...",  # or "video_path": "/path/to/video.mp4"
            "device": "cuda",  # optional, default: "cuda"
            "output_dir": "/path/to/output",  # optional
            "api_url": "https://api.example.com/api/passes",  # optional: API endpoint to send JSON data
            "api_headers": {"Authorization": "Bearer token"},  # optional: custom headers for API
            "video_id": "video_123"  # optional: video identifier for API
        }
    }
    """
    try:
        # Parse input
        input_data = event.get('input', {})
        video_url = input_data.get('video_url')
        video_path = input_data.get('video_path')
        device = input_data.get('device', 'cuda')
        output_dir = input_data.get('output_dir')
        api_url = input_data.get('api_url')  # Optional API endpoint
        api_headers = input_data.get('api_headers')  # Optional API headers
        video_id = input_data.get('video_id')  # Optional video ID
        
        # Ensure models are available
        ensure_models()
        
        # Get video file
        if video_url:
            video_path = '/tmp/input_video.mp4'
            print(f"Downloading video from {video_url}...")
            download_video(video_url, video_path)
        elif not video_path:
            return {
                "error": "Either 'video_url' or 'video_path' must be provided"
            }
        
        if not Path(video_path).exists():
            return {
                "error": f"Video file not found: {video_path}"
            }
        
        video_path = Path(video_path)
        print(f"Processing video: {video_path}")
        
        # Determine output directory
        if output_dir:
            analysis_dir = Path(output_dir)
        else:
            analysis_dir = video_path.parent / 'analysis' / video_path.stem
        
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Run telemetry logger
        print("Step 1: Generating telemetry data...")
        telemetry_path = analysis_dir / 'telemetry.jsonl'
        metadata_path = analysis_dir / 'metadata.json'
        
        # Import and run telemetry logger
        import sys
        sys.argv = [
            'telemetry_logger.py',
            '--source_video_path', str(video_path),
            '--output_dir', str(analysis_dir),
            '--device', device,
            '--sample_stride', '120'
        ]
        from tools.telemetry_logger import main as telemetry_main
        telemetry_main()
        
        if not telemetry_path.exists():
            return {
                "error": "Telemetry generation failed - no output file created"
            }
        
        print(f"✅ Telemetry generated: {telemetry_path}")
        
        # Step 2: Run pass detection
        print("Step 2: Detecting passes...")
        csv_path = analysis_dir / 'passes_from_telemetry.csv'
        json_path = analysis_dir / 'passes_from_telemetry.json'
        
        sys.argv = [
            'pass_events_from_telemetry.py',
            '--telemetry_path', str(telemetry_path),
            '--metadata_path', str(metadata_path),
            '--output_dir', str(analysis_dir),
            '--possession_radius_cm', '600.0',
            '--min_possession_frames', '2',
            '--pass_timeout_frames', '50'
        ]
        from tools.pass_events_from_telemetry import main as pass_main
        pass_main()
        
        # Read results
        results = {
            "status": "success",
            "video": str(video_path),
            "telemetry_file": str(telemetry_path),
            "metadata_file": str(metadata_path),
            "passes_csv": str(csv_path) if csv_path.exists() else None,
            "passes_json": str(json_path) if json_path.exists() else None,
        }
        
        # Add pass count if available
        if csv_path.exists():
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                results["total_passes"] = sum(1 for row in reader) - 1  # Subtract header
        
        # Read JSON if available
        if json_path.exists():
            with open(json_path, 'r') as f:
                results["passes_data"] = json.load(f)
        
        print(f"✅ Analysis complete! Found {results.get('total_passes', 0)} passes")
        
        # Step 3: Send data to API if API URL is provided
        if api_url and json_path.exists():
            print(f"Step 3: Sending pass data to API: {api_url}")
            try:
                # Prepare metadata
                metadata = {
                    "video_path": str(video_path),
                    "telemetry_file": str(telemetry_path),
                    "metadata_file": str(metadata_path),
                    "device": device
                }
                
                # Load metadata.json if available
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata["video_metadata"] = json.load(f)
                
                # Import here to ensure path is set
                from tools.api_client import send_pass_data_with_summary
                
                # Send to API
                api_result = send_pass_data_with_summary(
                    json_path=json_path,
                    api_url=api_url,
                    video_id=video_id or video_path.stem,
                    metadata=metadata,
                    headers=api_headers,
                    timeout=30
                )
                
                if api_result.get("success"):
                    results["api_status"] = "sent"
                    results["api_response"] = api_result.get("response", {})
                    results["api_summary"] = api_result.get("summary", {})
                    print(f"✅ Data sent to API successfully! Status: {api_result.get('status_code')}")
                else:
                    results["api_status"] = "failed"
                    results["api_error"] = api_result.get("error", "Unknown error")
                    print(f"⚠️  Failed to send data to API: {api_result.get('error')}")
                    
            except Exception as api_error:
                results["api_status"] = "error"
                results["api_error"] = str(api_error)
                print(f"⚠️  Error sending data to API: {str(api_error)}")
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"ERROR: {error_msg}")
        print(traceback_str)
        return {
            "error": error_msg,
            "traceback": traceback_str
        }


# RunPod serverless entry point
if __name__ == "__main__":
    # Read input from stdin (RunPod format)
    input_text = sys.stdin.read()
    try:
        event = json.loads(input_text)
    except:
        # If not JSON, try as plain text (for testing)
        event = {"input": {"video_path": input_text.strip()}}
    
    result = handler(event)
    print(json.dumps(result, indent=2))



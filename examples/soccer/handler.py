"""
RunPod Serverless Handler for Soccer Video Analysis
"""
import os
import json
import sys
import traceback
from pathlib import Path
import subprocess
import requests # Added for API calls

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
            "base_api_url": "https://app.wizard.net.co", # Required: Base URL for match API
            "match_id": "fb7e5817-9a84-44f1-bcb8-d5cfeca7d647", # Required: Match ID for API calls
            "api_key": "YOUR_API_KEY", # Required: API Key for authorization
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
        
        base_api_url = input_data.get('base_api_url')
        match_id = input_data.get('match_id')
        api_key = input_data.get('api_key')

        if not base_api_url or not match_id or not api_key:
            return {
                "error": "Missing required API parameters: base_api_url, match_id, or api_key"
            }
        
        api_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Step 0: Update match status to PROCESSING
        print(f"Step 0: Updating match {match_id} status to PROCESSING...")
        try:
            status_url = f"{base_api_url}/match/{match_id}"
            response = requests.post(status_url, headers=api_headers, json={"status": "PROCESSING"})
            response.raise_for_status()
            print(f"✅ Match status updated to PROCESSING. Response: {response.status_code}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to update match status to PROCESSING: {e}"
            print(f"⚠️  {error_msg}")
            return {"error": error_msg}

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
        
        # Step 3: Send pass results to API
        if json_path.exists():
            print(f"Step 3: Sending pass results to API: {base_api_url}/match/{match_id}/results")
            try:
                with open(json_path, 'r') as f:
                    pass_results = json.load(f)
                
                results_url = f"{base_api_url}/match/{match_id}/results"
                response = requests.post(results_url, headers=api_headers, json={"result": pass_results})
                response.raise_for_status()
                results["results_api_status"] = "sent"
                results["results_api_response"] = response.status_code
                print(f"✅ Pass results sent to API successfully! Status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                error_msg = f"Failed to send pass results to API: {e}"
                print(f"⚠️  {error_msg}")
                results["results_api_status"] = "failed"
                results["results_api_error"] = error_msg
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode passes_from_telemetry.json: {e}"
                print(f"⚠️  {error_msg}")
                results["results_api_status"] = "failed"
                results["results_api_error"] = error_msg

        # Step 4: Update match status to COMPLETED
        print(f"Step 4: Updating match {match_id} status to COMPLETED...")
        try:
            status_url = f"{base_api_url}/match/{match_id}"
            response = requests.post(status_url, headers=api_headers, json={"status": "COMPLETED"})
            response.raise_for_status()
            results["final_status_api_status"] = "sent"
            results["final_status_api_response"] = response.status_code
            print(f"✅ Match status updated to COMPLETED. Response: {response.status_code}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to update match status to COMPLETED: {e}"
            print(f"⚠️  {error_msg}")
            results["final_status_api_status"] = "failed"
            results["final_status_api_error"] = error_msg
        
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



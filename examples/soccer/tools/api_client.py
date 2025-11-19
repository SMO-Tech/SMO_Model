"""
API Client for sending pass analysis data to backend API
"""
import json
import requests
from typing import Dict, Optional, Any
from pathlib import Path


def send_pass_data_to_api(
    json_path: Path,
    api_url: str,
    video_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Send pass analysis JSON data to API endpoint via POST request.
    
    Args:
        json_path: Path to the passes_from_telemetry.json file
        api_url: API endpoint URL (e.g., "https://api.example.com/passes")
        video_id: Optional video identifier
        metadata: Optional additional metadata to include
        timeout: Request timeout in seconds
        headers: Optional custom headers (default: application/json)
    
    Returns:
        Dict with response data or error information
    
    Example:
        result = send_pass_data_to_api(
            json_path=Path("analysis/video/passes_from_telemetry.json"),
            api_url="https://api.example.com/api/passes",
            video_id="video_123",
            metadata={"source": "soccer_analysis"}
        )
    """
    if not json_path.exists():
        return {
            "success": False,
            "error": f"JSON file not found: {json_path}"
        }
    
    # Load pass data from JSON
    try:
        with open(json_path, 'r') as f:
            pass_data = json.load(f)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read JSON file: {str(e)}"
        }
    
    # Prepare request payload
    payload = {
        "passes": pass_data,
        "total_passes": len(pass_data)
    }
    
    if video_id:
        payload["video_id"] = video_id
    
    if metadata:
        payload["metadata"] = metadata
    
    # Prepare headers
    if headers is None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    # Send POST request
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        
        # Check response
        response.raise_for_status()
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response": response.json() if response.content else {},
            "message": "Data sent successfully"
        }
    
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timeout after {timeout} seconds"
        }
    
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP error: {str(e)}",
            "status_code": response.status_code if 'response' in locals() else None,
            "response": response.json() if 'response' in locals() and response.content else {}
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def send_pass_data_with_summary(
    json_path: Path,
    api_url: str,
    video_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Send pass data with summary statistics to API.
    
    Calculates summary statistics (total passes, successful, intercepted, lost,
    short/medium/long pass counts) and includes them in the payload.
    """
    if not json_path.exists():
        return {
            "success": False,
            "error": f"JSON file not found: {json_path}"
        }
    
    # Load pass data
    try:
        with open(json_path, 'r') as f:
            pass_data = json.load(f)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read JSON file: {str(e)}"
        }
    
    # Calculate summary statistics
    total_passes = len(pass_data)
    successful = sum(1 for p in pass_data if p.get("outcome") == "successful")
    intercepted = sum(1 for p in pass_data if p.get("outcome") == "intercepted")
    lost = sum(1 for p in pass_data if p.get("outcome") == "lost")
    
    short_passes = sum(1 for p in pass_data if p.get("pass_type") == "short")
    medium_passes = sum(1 for p in pass_data if p.get("pass_type") == "medium")
    long_passes = sum(1 for p in pass_data if p.get("pass_type") == "long")
    
    summary = {
        "total_passes": total_passes,
        "successful": successful,
        "intercepted": intercepted,
        "lost": lost,
        "short_passes": short_passes,
        "medium_passes": medium_passes,
        "long_passes": long_passes,
        "success_rate": (successful / total_passes * 100) if total_passes > 0 else 0
    }
    
    # Prepare payload
    payload = {
        "passes": pass_data,
        "summary": summary
    }
    
    if video_id:
        payload["video_id"] = video_id
    
    if metadata:
        payload["metadata"] = metadata
    
    # Send request
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        
        response.raise_for_status()
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response": response.json() if response.content else {},
            "summary": summary,
            "message": "Data sent successfully"
        }
    
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timeout after {timeout} seconds"
        }
    
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP error: {str(e)}",
            "status_code": response.status_code if 'response' in locals() else None,
            "response": response.json() if 'response' in locals() and response.content else {}
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


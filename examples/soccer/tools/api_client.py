"""
API Client for sending pass detection results to external API
"""
import json
import requests
from pathlib import Path
from typing import Dict, Optional, Any

# Default API base URL
DEFAULT_API_BASE_URL = "http://api.scoutme.cloud"


def update_match_status(
    match_id: str,
    status: str,
    base_url: str = DEFAULT_API_BASE_URL,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Update match status via API (call this before/after processing)
    
    Args:
        match_id: UUID of the match
        status: Status to set ("processing", "completed", "failed", etc.)
        base_url: API base URL (default: http://api.scoutme.cloud)
        headers: Custom HTTP headers (e.g., Authorization)
        timeout: Request timeout in seconds
    
    Returns:
        Dict with success status and response
    """
    try:
        api_url = f"{base_url.rstrip('/')}/match/{match_id}"
        
        payload = {
            "status": status
        }
        
        request_headers = {
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)
        
        print(f"[API] Updating match {match_id} status to '{status}'...")
        response = requests.post(
            api_url,
            json=payload,
            headers=request_headers,
            timeout=timeout
        )
        
        if response.status_code in [200, 201]:
            print(f"[API] ✅ Status updated successfully")
            return {
                "success": True,
                "status_code": response.status_code,
                "response": response.json() if response.content else {}
            }
        else:
            print(f"[API] ⚠️ Status update failed: {response.status_code}")
            return {
                "success": False,
                "status_code": response.status_code,
                "error": f"API returned status {response.status_code}: {response.text[:200]}"
            }
            
    except requests.exceptions.RequestException as e:
        print(f"[API] ❌ Request failed: {str(e)}")
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        print(f"[API] ❌ Unexpected error: {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def send_pass_data_with_summary(
    json_path: Path,
    api_url: str,
    video_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Send pass detection data to API endpoint with summary statistics
    
    Args:
        json_path: Path to passes_from_telemetry.json file
        api_url: API endpoint URL
        video_id: Unique identifier for the video
        metadata: Additional metadata to include
        headers: Custom HTTP headers (e.g., Authorization)
        timeout: Request timeout in seconds
    
    Returns:
        Dict with success status, response, and summary
    """
    try:
        # Load pass data
        if not json_path.exists():
            return {
                "success": False,
                "error": f"JSON file not found: {json_path}"
            }
        
        with open(json_path, 'r') as f:
            passes_data = json.load(f)
        
        # Calculate summary statistics
        total_passes = len(passes_data)
        successful = sum(1 for p in passes_data if p.get('outcome') == 'successful')
        intercepted = sum(1 for p in passes_data if p.get('outcome') == 'intercepted')
        lost = sum(1 for p in passes_data if p.get('outcome') == 'lost')
        
        short_passes = sum(1 for p in passes_data if p.get('pass_type') == 'short')
        medium_passes = sum(1 for p in passes_data if p.get('pass_type') == 'medium')
        long_passes = sum(1 for p in passes_data if p.get('pass_type') == 'long')
        
        # Calculate speed statistics (enhanced telemetry data)
        release_speeds = [p.get('release_speed') for p in passes_data if p.get('release_speed') is not None]
        receive_speeds = [p.get('receive_speed') for p in passes_data if p.get('receive_speed') is not None]
        distances = [p.get('distance_m') for p in passes_data if p.get('distance_m') is not None]
        
        # Count recovered passes (ball position was interpolated/refined)
        release_recovered = sum(1 for p in passes_data if p.get('release_recovered'))
        receive_recovered = sum(1 for p in passes_data if p.get('receive_recovered'))
        
        # Prepare payload
        payload = {
            "video_id": video_id,
            "summary": {
                "total_passes": total_passes,
                "successful": successful,
                "intercepted": intercepted,
                "lost": lost,
                "success_rate": successful / total_passes if total_passes > 0 else 0,
                "pass_types": {
                    "short": short_passes,
                    "medium": medium_passes,
                    "long": long_passes
                },
                "speed_stats": {
                    "avg_release_speed": sum(release_speeds) / len(release_speeds) if release_speeds else None,
                    "avg_receive_speed": sum(receive_speeds) / len(receive_speeds) if receive_speeds else None,
                    "max_release_speed": max(release_speeds) if release_speeds else None,
                    "max_receive_speed": max(receive_speeds) if receive_speeds else None,
                },
                "distance_stats": {
                    "avg_distance_m": sum(distances) / len(distances) if distances else None,
                    "max_distance_m": max(distances) if distances else None,
                    "min_distance_m": min(distances) if distances else None,
                },
                "recovery_stats": {
                    "release_recovered_count": release_recovered,
                    "receive_recovered_count": receive_recovered,
                }
            },
            "passes": passes_data,
            "metadata": metadata or {}
        }
        
        # Prepare headers
        request_headers = {
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)
        
        # Send POST request
        response = requests.post(
            api_url,
            json=payload,
            headers=request_headers,
            timeout=timeout
        )
        
        # Check response
        if response.status_code in [200, 201]:
            return {
                "success": True,
                "status_code": response.status_code,
                "response": response.json() if response.content else {},
                "summary": payload["summary"]
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": f"API returned status {response.status_code}: {response.text[:200]}",
                "response": response.text[:500] if response.text else None
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON in response: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

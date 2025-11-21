"""
API Client for sending pass detection results to external API
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import requests

class DummyAPIClient:
    def __init__(self, base_api_url: str, match_id: str, api_key: str):
        self.base_api_url = base_api_url
        self.match_id = match_id
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def update_match_status(self, status: str) -> Dict[str, Any]:
        print(f"\n--- DUMMY API CLIENT: update_match_status called (Status: {status}) ---")
        print(f"API URL: {self.base_api_url}/match/{self.match_id}")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")
        print(f"Payload: {{'status': '{status}'}}")
        print("--- END DUMMY API CLIENT ---\n")
        return {"success": True, "status_code": 200, "response": {"message": f"Status updated to {status}"}}

    def send_match_results(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- DUMMY API CLIENT: send_match_results called ---")
        print(f"API URL: {self.base_api_url}/match/{self.match_id}/results")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")
        print(f"Payload: {json.dumps(results_data, indent=2)}")
        print("--- END DUMMY API CLIENT ---\n")
        return {"success": True, "status_code": 200, "response": {"message": "Results sent"}}

# Original send_pass_data_with_summary (can be removed if not needed directly by handler)
def send_pass_data_with_summary(
    json_path: Path,
    api_url: str,
    video_id: str,
    metadata: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    # This function is now deprecated in favor of DummyAPIClient.send_match_results for testing
    # or the real API call in production.
    print("\n--- DUMMY API CLIENT: send_pass_data_with_summary called (DEPRECATED for handler direct use) ---")
    print(f"JSON Path: {json_path}")
    print(f"API URL: {api_url}")
    print(f"Video ID: {video_id}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    
    try:
        with open(json_path, 'r') as f:
            pass_data = json.load(f)
        print(f"Pass Data (from {json_path}): {json.dumps(pass_data, indent=2)}")
    except Exception as e:
        print(f"Failed to read pass data from {json_path}: {e}")
        
    print("--- END DUMMY API CLIENT (DEPRECATED) ---\n")
    
    return {
        "success": True,
        "status_code": 200,
        "response": {"message": "Dummy API call successful"},
        "summary": {"total_passes": 10, "successful_passes": 8}
    }

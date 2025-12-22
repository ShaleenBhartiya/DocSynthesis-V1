#!/usr/bin/env python3
"""
Example: Using the REST API
Demonstrates how to use the DocSynthesis-V1 REST API.
"""

import requests
import time
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def process_document_via_api(document_path: str):
    """Process a document using the REST API."""
    
    print(f"Processing document via API: {document_path}")
    
    # Step 1: Upload and start processing
    print("\n1. Uploading document...")
    
    with open(document_path, 'rb') as f:
        files = {'file': f}
        options = {
            'translate': True,
            'extract_fields': True,
            'generate_summary': True,
            'explain': True
        }
        data = {'options': json.dumps(options)}
        
        response = requests.post(
            f"{API_URL}/api/v1/process",
            files=files,
            data=data
        )
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    result = response.json()
    job_id = result['job_id']
    print(f"✅ Document uploaded. Job ID: {job_id}")
    
    # Step 2: Poll for status
    print("\n2. Waiting for processing to complete...")
    
    while True:
        response = requests.get(f"{API_URL}/api/v1/status/{job_id}")
        status_data = response.json()
        
        print(f"   Status: {status_data['status']} ({status_data['progress']}%)")
        
        if status_data['status'] == 'completed':
            print("✅ Processing completed!")
            break
        elif status_data['status'] == 'failed':
            print("❌ Processing failed!")
            return
        
        time.sleep(2)  # Wait 2 seconds before checking again
    
    # Step 3: Get results
    print("\n3. Retrieving results...")
    
    response = requests.get(f"{API_URL}/api/v1/results/{job_id}")
    
    if response.status_code != 200:
        print(f"❌ Failed to get results: {response.text}")
        return
    
    results = response.json()
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if results.get('ocr'):
        print(f"\nOCR Confidence: {results['ocr']['confidence']:.2%}")
    
    if results.get('extraction'):
        print(f"\nExtracted Fields:")
        for field, value in list(results['extraction']['fields'].items())[:5]:
            print(f"  - {field}: {value}")
    
    if results.get('summary'):
        print(f"\nSummary:")
        print(f"  {results['summary']['short'][:200]}...")
    
    print("\n" + "="*80)

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/api/v1/health")
        if response.status_code == 200:
            print("✅ API is healthy and ready")
            return True
        else:
            print("❌ API is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print(f"   Start the server with: python -m src.api.server")
        return False

def main():
    print("DocSynthesis-V1 API Client Example")
    print("="*80)
    
    # Check API health
    print("\nChecking API health...")
    if not check_api_health():
        return
    
    # Process a document
    document_path = "examples/sample_certificate.pdf"
    
    if not Path(document_path).exists():
        print(f"\n❌ Sample document not found: {document_path}")
        print("Please place a sample document in the examples/ directory")
        return
    
    process_document_via_api(document_path)

if __name__ == "__main__":
    main()


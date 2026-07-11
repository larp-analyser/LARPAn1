import urllib.request
import urllib.error
import json
import time

TARGET_URL = "https://larpan1.onrender.com/an1"

payload = {
    "message": "Hey @an1, what do you think about @victim_user?",
    "sender_id": "usr_t1",
    "username": "test_user1",
    "display_name": "Test User 1",
    "group_name": "test_group",
    "channel": "general",
    "platform": "discord",
    "force_reply": True,
    "mode": "legacy"
}

def main():
    print("="*60)
    print(" LARPAn1 Single Payload Tester ")
    print(f" Target: {TARGET_URL}")
    print("="*60)
    
    print("\nSending Payload:")
    print(json.dumps(payload, indent=2))
    
    start_time = time.time()
    
    req = urllib.request.Request(
        TARGET_URL, 
        data=json.dumps(payload).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}, 
        method='POST'
    )
    
    try:
        print("\nWaiting for response... (Render cold starts may take ~50s)")
        with urllib.request.urlopen(req, timeout=90.0) as response:
            elapsed = time.time() - start_time
            print(f"\n\033[92m[ SUCCESS ] Response received in {elapsed:.2f}s\033[0m")
            print("Response Data:")
            print(json.dumps(json.loads(response.read().decode('utf-8')), indent=2))
            
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        print(f"\n\033[91m[ FAILED ] HTTP {e.code} ({elapsed:.2f}s)\033[0m")
        print(e.read().decode('utf-8') or str(e))
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n\033[91m[ ERROR ] Request failed ({elapsed:.2f}s)\033[0m")
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    main()

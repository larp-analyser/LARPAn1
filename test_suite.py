import urllib.request
import urllib.error
import json
import time

TARGET_URL = "https://larpan1.onrender.com/an1"

payload = {
    "message": "what happened yo @an1?",
    "sender_id": "002",
    "username": "alice",
    "display_name": "alicename",
    # Where the message was sent. Use "twitter_public", "discord_dm", "general", etc.
    "group_name": "twitter_public",
    "channel": "timeline",
    "platform": "twitter",
    
    # Array of users tagged in the message. This triggers the tagged profiles extraction feature!
    # Example: [{"id": "999", "username": "fake_ceo", "display_name": "Fake CEO"}]
    # Leave as an empty list [] if no one is tagged.
    "tagged_users": [],
        #{
        #    "id": "99999",
        #    "username": "fake_ceo",
        #   "display_name": "Fake CEO"
        #}
    #],
    
    # If True, bypasses the Triage Gatekeeper and forces the bot to generate a reply.
    # If False, the bot might choose to remain silent (return None) if the message is boring.
    "force_reply": False,
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

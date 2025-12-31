import requests
import os
import json
import dotenv

dotenv.load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

def auth():
    r = requests.get(f"https://www.deviantart.com/oauth2/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}")
    return json.loads(r.text)['access_token']

access_token = auth()


header = {
    "Authorization": f"Bearer {access_token}"
}

resp = requests.get(" https://www.deviantart.com/api/v1/oauth2/browse/tags?tag=cats",headers=header)

resp.raise_for_status()
data = resp.json()

print(data)



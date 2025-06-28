import requests
import json

response = requests.get('https://api.blocknative.com/gasprices/blockprices')
print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))
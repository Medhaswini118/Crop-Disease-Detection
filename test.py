import requests

url = "http://127.0.0.1:5000/predict"
file_path = r"E:\crop-disease-detection\test\AppleCedarRust3.JPG"  # Use raw string (r"")

files = {"file": open(file_path, "rb")}
response = requests.post(url, files=files)

print(response.json())

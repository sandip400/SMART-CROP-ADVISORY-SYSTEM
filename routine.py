import requests

API_KEY = "sk-FCGh68ac7c0bc7bf312038"
BASE_URL = "https://perenual.com/api/v2"

def fetch_plant_care(common_name):
    # Step 1: Search for plant ID
    search_url = f"{BASE_URL}/species-list"
    params = {"key": API_KEY, "q": common_name}
    resp = requests.get(search_url, params=params)
    data = resp.json().get("data", [])
    if not data:
        return None

    plant_id = data[0]["id"]

    # Step 2: Fetch detailed care info
    details_url = f"{BASE_URL}/species/details/{plant_id}"
    resp2 = requests.get(details_url, params={"key": API_KEY})
    details = resp2.json()

    care_info = {
        "common_name": details.get("common_name"),
        "watering_frequency": details.get("watering_general_benchmark"),
        "watering_type": details.get("watering"),
        "sunlight": details.get("sunlight"),
        # Add more fields if needed
    }
    return care_info

if __name__ == "__main__":
    for plant in ["Tomato", "Rose", "Groundnut"]:
        info = fetch_plant_care(plant)
        print(plant, "->", info)

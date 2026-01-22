import requests

# Replace with your USDA API key
API_KEY = "XWVTX6rRelxDoLlglqtA9WkP8s07kL4d2mdcwmGJ"
food_query = "pizza"

# USDA FoodData Central API endpoint
url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_query}&api_key={API_KEY}"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    if data['foods']:
        food = data['foods'][0]
        print("Food:", food['description'])
        print("Nutrients:", [(n['nutrientName'], n['value'], n['unitName']) for n in food.get('foodNutrients', [])][:3])  # Top 3 nutrients
    else:
        print("No food found.")
else:
    print("Error:", response.status_code)
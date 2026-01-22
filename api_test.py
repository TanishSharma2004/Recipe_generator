import requests

API_KEY = "474f2418c0904d18a2ab6d3bf436a411"

dish_name = "strawberry cheesecake"

search_url = f"https://api.spoonacular.com/recipes/complexSearch?query={dish_name}&apiKey={API_KEY}"
response = requests.get(search_url)

if response.status_code == 200:
    data = response.json()
    if data['results']:
        recipe = data['results'][0]
        recipe_id = recipe['id']
        print("Recipe Title:", recipe['title'])
        
        details_url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?includeNutrition=false&apiKey={API_KEY}"
        details_response = requests.get(details_url)
        
        if details_response.status_code == 200:
            details = details_response.json()
            print("Ingredients:")
            for ingredient in details.get('extendedIngredients', []):
                print(f"- {ingredient['original']}")
        else:
            print("Error fetching details:", details_response.status_code)
    else:
        print("No recipes found.")
else:
    print("Error searching:", response.status_code)
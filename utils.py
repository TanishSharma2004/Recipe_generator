import requests
import json
import re

# API Keys - Free tier limits considered
SPOONACULAR_API_KEY = "474f2418c0904d18a2ab6d3bf436a411"  # Free: 150 requests/day
USDA_API_KEY = "XWVTX6rRelxDoLlglqtA9WkP8s07kL4d2mdcwmGJ"  # Free: 1000 requests/hour

# Cache to reduce API calls (important for free tier!)
recipe_cache = {}
nutrition_cache = {}

def get_recipe_from_spoonacular(dish_name):
    """Fetch recipe from Spoonacular API with caching"""
    # Check cache first
    if dish_name in recipe_cache:
        print(f"[Cache] Using cached recipe for {dish_name}")
        return recipe_cache[dish_name]
    
    try:
        # Search for recipe
        search_url = f"https://api.spoonacular.com/recipes/complexSearch?query={dish_name}&number=1&apiKey={SPOONACULAR_API_KEY}"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                recipe = data['results'][0]
                recipe_id = recipe['id']
                
                # Get detailed recipe information
                details_url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?includeNutrition=true&apiKey={SPOONACULAR_API_KEY}"
                details_response = requests.get(details_url, timeout=10)
                
                if details_response.status_code == 200:
                    details = details_response.json()
                    
                    # Extract ingredients
                    ingredients = [ing['original'] for ing in details.get('extendedIngredients', [])]
                    
                    # Extract instructions
                    instructions = []
                    if 'analyzedInstructions' in details and details['analyzedInstructions']:
                        for instruction_set in details['analyzedInstructions']:
                            for step in instruction_set.get('steps', []):
                                instructions.append(f"{step['number']}. {step['step']}")
                    elif 'instructions' in details and details['instructions']:
                        # Clean HTML tags
                        clean_instructions = re.sub('<.*?>', '', details['instructions'])
                        instructions = [clean_instructions]
                    
                    # Extract nutrition
                    nutrition_data = {}
                    if 'nutrition' in details and 'nutrients' in details['nutrition']:
                        for nutrient in details['nutrition']['nutrients'][:10]:
                            nutrition_data[nutrient['name']] = {
                                'amount': nutrient['amount'],
                                'unit': nutrient['unit']
                            }
                    
                    result = {
                        'title': details.get('title', dish_name),
                        'ingredients': ingredients,
                        'instructions': instructions,
                        'nutrition': nutrition_data,
                        'servings': details.get('servings', 'N/A'),
                        'ready_in_minutes': details.get('readyInMinutes', 'N/A'),
                        'source_url': details.get('sourceUrl', '')
                    }
                    
                    # Cache the result
                    recipe_cache[dish_name] = result
                    return result
        elif response.status_code == 402:
            print("⚠️ Spoonacular API limit reached (150/day). Try again tomorrow!")
            return None
        
        return None
    except Exception as e:
        print(f"Error fetching from Spoonacular: {e}")
        return None

def get_nutrition_from_usda(food_name):
    """Fetch nutrition data from USDA API with caching"""
    # Check cache first
    if food_name in nutrition_cache:
        print(f"[Cache] Using cached nutrition for {food_name}")
        return nutrition_cache[food_name]
    
    try:
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&pageSize=1&api_key={USDA_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('foods'):
                food = data['foods'][0]
                nutrients = {}
                
                for nutrient in food.get('foodNutrients', [])[:15]:
                    nutrients[nutrient['nutrientName']] = {
                        'value': round(nutrient.get('value', 0), 2),
                        'unit': nutrient['unitName']
                    }
                
                result = {
                    'description': food['description'],
                    'nutrients': nutrients
                }
                
                # Cache the result
                nutrition_cache[food_name] = result
                return result
        
        return None
    except Exception as e:
        print(f"Error fetching from USDA: {e}")
        return None

def get_fun_facts(dish_name):
    """Generate fun facts about the dish"""
    fun_facts_db = {
        'pizza': [
            "🍕 The world's most expensive pizza costs $12,000 and takes 72 hours to make!",
            "🍕 Americans eat approximately 350 slices of pizza per second.",
            "🍕 October is National Pizza Month in the USA."
        ],
        'biryani': [
            "🍛 Biryani originated in Persia and was brought to India by Mughal rulers.",
            "🍛 There are over 26 varieties of biryani in India alone!",
            "🍛 The word 'biryani' comes from Persian 'birian' meaning 'fried before cooking'."
        ],
        'burger': [
            "🍔 The largest burger ever made weighed 2,014 pounds!",
            "🍔 Americans consume about 50 billion burgers per year.",
            "🍔 The hamburger got its name from Hamburg, Germany."
        ],
        'sushi': [
            "🍣 Sushi was originally a way to preserve fish in fermented rice.",
            "🍣 Wasabi's green color naturally repels bacteria.",
            "🍣 The most expensive sushi can cost over $2,000 per piece!"
        ],
        'pasta': [
            "🍝 There are over 600 different shapes of pasta produced worldwide.",
            "🍝 Italians eat pasta an average of 23.5 kg per person per year.",
            "🍝 Marco Polo didn't bring pasta to Italy - it was already there!"
        ],
        'chicken': [
            "🍗 Chicken is the most common type of poultry in the world.",
            "🍗 A chicken can run up to 9 mph (14 km/h)!",
            "🍗 There are more chickens on Earth than people."
        ],
        'cake': [
            "🎂 The most expensive cake ever made cost $75 million!",
            "🎂 Ancient Egyptians were the first to show evidence of baking skills.",
            "🎂 The tradition of birthday cakes dates back to ancient Greece."
        ],
        'ice cream': [
            "🍦 It takes about 50 licks to finish one scoop of ice cream.",
            "🍦 Vanilla is the most popular ice cream flavor worldwide.",
            "🍦 The ice cream cone was invented in 1904 at the World's Fair."
        ],
        'salad': [
            "🥗 The word 'salad' comes from the Latin word 'salata' meaning salted.",
            "🥗 Caesar salad was invented in Mexico, not Italy!",
            "🥗 Eating salads can improve your skin health and complexion."
        ],
        'soup': [
            "🍲 Soup has been made since ancient times - over 20,000 years ago!",
            "🍲 Chicken soup is called 'Jewish penicillin' for its healing properties.",
            "🍲 The largest soup bowl ever made held 6,000 gallons!"
        ],
        'default': [
            "🍽️ Food tastes better when eaten with others!",
            "🍽️ The average person eats about 35 tons of food in a lifetime.",
            "🍽️ Cooking at home is healthier and more economical than eating out.",
            "🍽️ Your sense of smell accounts for 80% of your taste!",
            "🍽️ Dark chocolate can improve your mood by releasing endorphins."
        ]
    }
    
    dish_lower = dish_name.lower()
    
    # Find matching category
    for key in fun_facts_db.keys():
        if key in dish_lower:
            return fun_facts_db[key]
    
    return fun_facts_db['default']

def combine_nutrition_data(spoonacular_data, usda_data):
    """Combine nutrition data from both APIs"""
    combined = {}
    
    if spoonacular_data and 'nutrition' in spoonacular_data:
        combined.update(spoonacular_data['nutrition'])
    
    if usda_data and 'nutrients' in usda_data:
        for nutrient, data in usda_data['nutrients'].items():
            if nutrient not in combined:
                combined[nutrient] = data
    
    return combined

def clear_cache():
    """Clear the API cache (call this if you want fresh data)"""
    global recipe_cache, nutrition_cache
    recipe_cache.clear()
    nutrition_cache.clear()
    print("Cache cleared!")

# API usage tracking (for free tier monitoring)
def get_cache_stats():
    """Get cache statistics to monitor API usage"""
    return {
        'recipes_cached': len(recipe_cache),
        'nutrition_cached': len(nutrition_cache),
        'total_cached': len(recipe_cache) + len(nutrition_cache)
    }
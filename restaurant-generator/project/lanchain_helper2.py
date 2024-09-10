from secret_key import OPENAI_API_KEY
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# print(os.getenv('OPENAI_API_KEY'))

llm = OpenAI(temperature=0.7)


def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template='I want to open a restaurant for {cuisine} food. Could you suggest a fancy name for this?'
    )

    # Generate prompt for the restaurant name and get the response
    restaurant_name_prompt = prompt_template_name.format(cuisine=cuisine)
    restaurant_name = llm(restaurant_name_prompt).strip()
    print("Generated Restaurant Name:", restaurant_name)

    # Chain 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template='Suggest some menu items for {restaurant_name}. Return it as a comma-separated list'
    )

    # Generate prompt for the menu items and get the response
    menu_items_prompt = prompt_template_items.format(restaurant_name=restaurant_name)
    menu_items = llm(menu_items_prompt).strip()
    print("Generated Menu Items:", menu_items)

    # Return the combined response
    return {
        'cuisine': cuisine,
        'restaurant_name': restaurant_name,
        'menu_items': menu_items
    }


if __name__ == "__main__":
    result = generate_restaurant_name_and_items('Italian')
    print(result)

# This file also works

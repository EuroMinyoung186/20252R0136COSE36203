import json
import argparse
from openai import OpenAI
from tqdm import tqdm


prompt = '''
Given the situation description, generate a fashion recommendation for an outfit that suits the situation.

### Example Input:
Situation : I'm attending a casual picnic in the park with my family.
Recommendation : Pair your long cotton plaid bottoms with a simple solid top and comfortable sneakers for a relaxed family picnic look.

Situation : I'm meeting friends for a cozy coffee date on a cool afternoon.
Recommendation : Wear your striped cotton lapel-neck sweater with your solid cotton long pants for a warm, relaxed look—perfect for a cozy afternoon coffee date with friends.

### Input:
Situation : {situation}
Recommendation :
'''

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--situation", type=str)

    return parser.parse_args()

def get_prompt(situation, caption=""):
    return prompt.format(situation=situation)

def get_situation_data(client, situation):

    prompt_text = get_prompt(situation)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
    )
    recommendation = response.choices[0].message.content.strip()

    return recommendation

if __name__ == "__main__":
    args = build_args()
    client = OpenAI(api_key=args.api_key)
    recommendation = get_situation_data(client, args.situation_path)

    print(recommendation)


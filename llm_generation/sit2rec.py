import json
import argparse
from openai import OpenAI
from tqdm import tqdm


prompt = '''
Given the situation description, generate a fashion recommendation for an outfit that suits the situation.

### Example Input:
Caption : The lower clothing is of long length. The fabric is cotton and it has plaid patterns.
Situation : I'm attending a casual picnic in the park with my family.
Recommendation : Pair your long cotton plaid bottoms with a simple solid top and comfortable sneakers for a relaxed family picnic look.

Caption : His sweater has long sleeves, cotton fabric and stripe patterns. The neckline of it is lapel. The gentleman wears a long pants. The pants are with cotton fabric and solid color patterns.
Situation : I'm meeting friends for a cozy coffee date on a cool afternoon.
Recommendation : Wear your striped cotton lapel-neck sweater with your solid cotton long pants for a warm, relaxed look—perfect for a cozy afternoon coffee date with friends.

### Input:
Caption : {caption}
Situation : {situation}
Recommendation :
'''

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--situation_path", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()

def get_prompt(situation, caption=""):
    return prompt.format(situation=situation, caption=caption)

def get_situation_data(client, situation_path, batch_input):

    with open(situation_path, 'r', encoding='utf-8') as f:
        situation_data = json.load(f)

    with open(batch_input, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)

    try:
        with open('temp_sit2rec_output.json', 'r', encoding='utf-8') as f:
            existing = json.load(f)
    except:
        existing = []
    

    img_caption_situations = {}
    img_caption_situations.update(existing)

    for custom_id, item in tqdm(situation_data.items()):
        if custom_id in img_caption_situations:
            continue
        content = item['situation']
        caption = item['caption']

        prompt_text = get_prompt(content, caption=caption)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
        )
        recommendation = response.choices[0].message.content.strip()
        img_caption_situations[custom_id] = {
            "caption": caption,
            "situation": content,
            "recommendation": recommendation
        }

        with open('temp_sit2rec_output.json', 'w', encoding='utf-8') as temp_f:
            json.dump(img_caption_situations, temp_f, ensure_ascii=False, indent=2)
        
    return img_caption_situations

if __name__ == "__main__":
    args = build_args()
    client = OpenAI(api_key=args.api_key)
    img_caption_situations = get_situation_data(client, args.situation_path, args.caption_path)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(img_caption_situations, f, ensure_ascii=False, indent=2)

    print(f"Saved recommendations to {args.output_path}")


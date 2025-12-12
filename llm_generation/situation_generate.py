import json
import openai
from openai import OpenAI
from tqdm import tqdm
import argparse

prompt = """
You are a fashion designer.

Given a clothing description, generate one natural situation in which the outfit would be appropriate.
Output one and only one sentence.

Example:
Cloth Caption: His tank top has sleeves cut off, cotton fabric and pure color patterns. The neckline of it is round. The pants this man wears is of long length. The pants are with cotton fabric and pure color patterns.
Situation: I'm going out for some light exercise with my friends.

Cloth Caption: This guy wears a long-sleeve shirt with solid color patterns and a long trousers. The shirt is with cotton fabric. The neckline of the shirt is round. The trousers are with cotton fabric and solid color patterns.
Situation: I'll do a slow jog in the park with my dog.

Input:
Cloth Caption: {caption}
Situation:
"""

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = build_args()
    client = OpenAI(api_key=args.api_key)
    caption_path = args.caption_path

    img_caption_situations = {}

    with open(caption_path, 'r') as f:
        captions = json.load(f)

    for idx, (img_name, caption) in enumerate(tqdm(captions.items())):
        formatted_prompt = prompt.format(caption=caption)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt}
            ]
        )
        situation = response.choices[0].message.content.strip()
        img_caption_situations[img_name] = {
            "caption": caption,
            "situation": situation
        }
        
        if idx % 100 == 0:

            with open(args.output_path, 'w') as f:
                json.dump(img_caption_situations, f, indent=4)

    with open(args.output_path, 'w') as f:
        json.dump(img_caption_situations, f, indent=4)

        

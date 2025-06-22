import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize translation model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
translation_tokenizer = AutoTokenizer.from_pretrained("../Helsinki-NLP/opus-mt-en-zh")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("../Helsinki-NLP/opus-mt-en-zh").to(device)


def translate_en_to_zh(root_dir, input_file, output_file):
    """
    Translate English image captions to Chinese and save the result to a new JSON file.

    Args:
        root_dir (str): Directory containing the input and output files.
        input_file (str): Name of the input JSON file.
        output_file (str): Name of the output JSON file to save translated data.
    """
    input_path = os.path.join(root_dir, input_file)
    output_path = os.path.join(root_dir, output_file)

    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Translate captions with progress bar
    for key, value in tqdm(data.items(), desc="Translating captions"):
        caption = value.get('img-to-text_en', "")
        if caption and not value.get('img-to-text_zh'):  # Avoid re-translating
            try:
                inputs = translation_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = translation_model.generate(**inputs, max_length=128)
                translated_caption = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                value['img-to-text_zh'] = translated_caption
            except Exception as e:
                print(f"Translation failed for key: {key}, reason: {e}")
                value['img-to-text_zh'] = ""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save updated data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nTranslation complete. Output saved to: {output_path}")


if __name__ == '__main__':
    root_dir = '../../data/Weibo/'
    input_file = 'dataset_items_merged.json'
    output_file = 'output_translated.json'
    translate_en_to_zh(root_dir, input_file, output_file)

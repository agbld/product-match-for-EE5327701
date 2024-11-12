import torch
import numpy as np
import pandas as pd
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

def get_semantic_model():
    model = ORTModelForFeatureExtraction.from_pretrained('clw8998/semantic_model-for-EE5327701')
    tokenizer = AutoTokenizer.from_pretrained('clw8998/semantic_model-for-EE5327701')
    return model, tokenizer

def inference(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    parser = argparse.ArgumentParser(description='Product Matching between two CSV files')
    parser.add_argument('csv_file1', type=str, help='Path to the first CSV file')
    parser.add_argument('csv_file2', type=str, help='Path to the second CSV file')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold value for similarity (e.g., 0.9)')
    args = parser.parse_args()

    # Load CSV files
    df1 = pd.read_csv(args.csv_file1)
    df2 = pd.read_csv(args.csv_file2)

    # Ensure 'name' column exists
    if 'name' not in df1.columns or 'name' not in df2.columns:
        print("Both CSV files must contain a 'name' column.")
        return

    product_names1 = df1['name'].astype(str).tolist()
    product_names2 = df2['name'].astype(str).tolist()

    # Load semantic model
    model, tokenizer = get_semantic_model()

    # Compute embeddings
    print("Computing embeddings for the first CSV file...")
    embeddings1 = inference(tokenizer, model, product_names1)
    print("Computing embeddings for the second CSV file...")
    embeddings2 = inference(tokenizer, model, product_names2)

    # Match products
    print("Matching products...")
    all_products = []

    for name1, emb1 in tqdm(zip(product_names1, embeddings1), total=len(product_names1), desc='Matching Products'):
        max_similarity = 0
        best_match = None
        for name2, emb2 in zip(product_names2, embeddings2):
            similarity = cosine_similarity(emb1, emb2)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name2
        if max_similarity >= args.threshold:
            all_products.append((name1, best_match, max_similarity))
        else:
            all_products.append((name1, None, None))  # No match found

    # Separate matched and unmatched products
    matched_products = [item for item in all_products if item[1] is not None]
    unmatched_products = [item for item in all_products if item[1] is None]

    # Create DataFrame
    final_df = pd.DataFrame(matched_products + unmatched_products, columns=['product_name_1', 'product_name_2', 'similarity'])

    # Sort DataFrame: matched products first
    final_df['similarity'] = final_df['similarity'].fillna(0)
    final_df = final_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    final_df['similarity'] = final_df['similarity'].replace(0, np.nan)

    # Save the result to a CSV file
    output_filename = './output/matched_products.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"Matching completed. Results saved to {output_filename}.")

if __name__ == '__main__':
    main()

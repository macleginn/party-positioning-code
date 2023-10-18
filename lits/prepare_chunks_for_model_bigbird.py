import os
import json
from collections import defaultdict
import pandas as pd
from transformers import BigBirdTokenizerFast
from tqdm.auto import tqdm
import utils

chunk_file_path = '../data/chunks_for_manifestos_bigbird.json'
rile_path = '../data/RILE_by_chunk_bigbird.json'

print('Loading the data...')
data = pd.read_csv('../data/full_data_w_translations_cleaned.csv')
print('Preparing the data...')
normalised_labels = data.label.map(utils.normalise_label)
del data['label']
data.insert(0, 'label', normalised_labels)
manifesto_ids = data.apply(utils.get_manifesto_id, axis=1)
data.insert(0, 'manifesto_id', manifesto_ids)

model_name = 'google/bigbird-roberta-base'
max_chunk_size = 4095
tokeniser = BigBirdTokenizerFast.from_pretrained(model_name)

chunks_by_manifesto = defaultdict(list)
riles_by_manifesto = defaultdict(list)  # Each chunk has its own RILE score
for manifesto_id in tqdm(data.manifesto_id.unique(), desc='Splitting into chunks and computing RILEs'):
    sentences = data.loc[data.manifesto_id == manifesto_id].text_translated.to_list()
    labels = data.loc[data.manifesto_id == manifesto_id].label.to_list()
    chunk_buffer = []
    chunk_token_count = 0
    label_buffer = []
    for sentence, label in zip(sentences, labels):
        new_tokens = tokeniser.tokenize(sentence)
        n_tokens = len(new_tokens)
        if chunk_token_count + n_tokens <= max_chunk_size:
            chunk_buffer.append(sentence)
            chunk_token_count += n_tokens
            label_buffer.append(label)
        else:
            chunks_by_manifesto[manifesto_id].append(
                ' '.join(chunk_buffer))
            chunk_buffer.clear()
            chunk_token_count = 0
            riles_by_manifesto[manifesto_id].append(
                utils.compute_rile_from_list(label_buffer))
            label_buffer.clear()
    if chunk_token_count >= 1000:
        chunks_by_manifesto[manifesto_id].append(' '.join(chunk_buffer))
        riles_by_manifesto[manifesto_id].append(
            utils.compute_rile_from_list(label_buffer))
with open(chunk_file_path, 'w') as out:
    json.dump(chunks_by_manifesto, out, indent=2)
with open(rile_path, 'w') as out:
    json.dump(riles_by_manifesto, out, indent=2)

import json
import os
from math import ceil
from random import shuffle, seed

import pandas as pd
import torch
from scipy.stats import spearmanr

from transformers import BigBirdTokenizerFast, BigBirdModel
from tqdm.auto import tqdm

import utils


class RILERegressionHead(torch.nn.Module):
    """
    Computes the RILE score based on the embedding of
    a manifesto chunk computed by a long-input transformer
    model.
    """

    def __init__(self, input_dim, inner_dim=1024):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        # Squeeze the output between -1 and 1
        return torch.tanh(x)


def train_epoch(epoch_n, data, batch_size,
                tokeniser, encoder, regression_head,
                optimiser):
    encoder.train()
    n_train_steps = ceil(data.shape[0] / batch_size)
    losses = torch.zeros(n_train_steps)
    predictions = []
    for step in tqdm(range(n_train_steps), 
                     desc=f'Epoch {epoch_n+1}, training', 
                     leave=False):
        optimiser.zero_grad()
        lo = step * batch_size
        hi = lo + batch_size
        batch = data.text[lo:hi].to_list()
        tokens = tokeniser(batch, truncation=True,
                           padding='longest', return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokens.items()}
        cls_embeddings = encoder(**model_inputs).last_hidden_state[:, 0, :]
        riles_predicted = regression_head(cls_embeddings)
        riles_gold = torch.tensor(data.RILE[lo:hi].values).cuda()
        loss = torch.mean(torch.square(riles_predicted - riles_gold))
        loss.backward()
        optimiser.step()
        losses[step] = loss.item()
        predictions.extend(riles_predicted.detach().cpu().flatten().tolist())
    return losses.mean().item(), predictions


def validate_epoch(epoch_n, data, batch_size,
                   tokeniser, encoder, regression_head):
    encoder.eval()
    n_train_steps = ceil(data.shape[0] / batch_size)
    predictions = []
    for step in tqdm(range(n_train_steps), 
                     desc=f'Epoch {epoch_n+1}, validation', 
                     leave=False):
        lo = step * batch_size
        hi = lo + batch_size
        batch = data.text[lo:hi].to_list()
        tokens = tokeniser(batch, truncation=True,
                           padding='longest', return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            cls_embeddings = encoder(**model_inputs).last_hidden_state[:, 0, :]
            riles_predicted = regression_head(cls_embeddings)
        predictions.extend(riles_predicted.detach().cpu().flatten().tolist())
    return spearmanr(predictions, data.RILE).correlation, predictions


def main():
    # Preprocessing done in another script:
    # -- split translated manifestos into chunk of length less than 4096
    #    (this operation is dependent on the tokeniser/model)
    # -- compute RILEs for manifesto chunks

    seed(42)
    torch.manual_seed(42)

    chunk_file_path = '../data/chunks_for_manifestos_bigbird.json'
    rile_path = '../data/RILE_by_chunk_bigbird.json'

    with open(rile_path) as inp:
        rile_by_chunk = json.load(inp)

    model_name = 'google/bigbird-roberta-base'
    tokeniser = BigBirdTokenizerFast.from_pretrained(model_name)

    with open(chunk_file_path) as inp:
        chunks_by_manifesto = json.load(inp)

    dataset_path = '../data/bigbird_chunk_training_data.csv'
    if os.path.exists(dataset_path):
        all_data = pd.read_csv(dataset_path, sep='\t')
    else:
        # Construct a dataset mapping chunks to their precomputed RILE scores
        records = []
        for manifesto_id, chunks in tqdm(list(chunks_by_manifesto.items()), 
                                        desc='Matching chunks to RILEs', 
                                        leave=False):
            chunk_riles = rile_by_chunk[manifesto_id]
            assert len(chunks) == len(
                chunk_riles
            ), f'The number of RILE scores does not match the number of chunks for {manifesto_id}'
            for chunk, chunk_rile in zip(chunks, chunk_riles):
                records.append([manifesto_id, chunk, chunk_rile])
        all_data = pd.DataFrame.from_records(
            records, columns=['manifesto_id', 'text', 'RILE'])
        all_data.to_csv(dataset_path, sep='\t')
    
    print(f'Dataset size: {all_data.shape[0]}')
    n_epochs = 5
    batch_size = 4

    # Results by country
    results_dict = {}

    for (country, 
         data_train_all, 
         data_test) in utils.split_chunk_data_loco(all_data):
        indices = list(range(data_train_all.shape[0]))
        shuffle(indices)
        dev_size = len(indices) // 10
        dev_indices = indices[: dev_size]
        train_indices = indices[dev_size :]
        data_train = data_train_all.iloc[train_indices, :]
        data_dev = data_train_all.iloc[dev_indices, :]
        print(f'Splits sizes for {country}: train = {data_train.shape[0]}, '
            f'dev = {data_dev.shape[0]}, test = {data_test.shape[0]}')

        encoder = BigBirdModel.from_pretrained(model_name)
        encoder.cuda()
        regression_head = RILERegressionHead(input_dim=768)
        regression_head.cuda()
        optimiser = torch.optim.AdamW(
            list(encoder.parameters()) + list(regression_head.parameters()), 
            lr=10**(-5))

        best_dev_correlation = 0.0
        test_correlation = 0.0
        test_predictions = []
        dev_predictions = {}

        # For testing
        # data_train = data_train.iloc[:100, :]
        # data_dev = data_dev.iloc[:100, :]
        # data_test = data_test.iloc[:100, :]
        for epoch_n in tqdm(range(n_epochs), desc='Epochs', leave=False):
            epoch_train_loss, _ = train_epoch(
                epoch_n,
                data_train,
                batch_size,
                tokeniser,
                encoder,
                regression_head,
                optimiser)
            print(f'Epoch {epoch_n+1} loss: {epoch_train_loss}')

            epoch_dev_correlation, epoch_dev_predictions = validate_epoch(
                epoch_n,
                data_dev,
                batch_size,
                tokeniser,
                encoder,
                regression_head)
            print(
                f'Epoch {epoch_n+1} dev Spearman correlation: {epoch_dev_correlation}')
            dev_predictions[epoch_n] = epoch_dev_predictions

            epoch_test_correlation, epoch_test_predictions = validate_epoch(
                epoch_n,
                data_test,
                batch_size,
                tokeniser,
                encoder,
                regression_head)

            if epoch_dev_correlation > best_dev_correlation:
                best_dev_correlation = epoch_dev_correlation
                test_correlation = epoch_test_correlation
                test_predictions = epoch_test_predictions
    
        results_dict[country] = {
            'test': {
                'manifesto_ids': data_test.manifesto_id.to_list(),
                'predictions': test_predictions
            }
        }
    
        print(
            f'{country}, test Spearman correlation after {n_epochs} epochs: {test_correlation}')
        with open('results_bigbird_loco.json', 'w') as out:
            json.dump(results_dict, out)


if __name__ == '__main__':
    main()

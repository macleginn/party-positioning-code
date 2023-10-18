import json
import os
from math import ceil
from random import shuffle, seed

import pandas as pd
import torch

from transformers import LongformerTokenizerFast, LongformerModel
from tqdm.auto import tqdm

import utils

N_GPUS = 8


class RILEBinnedRegressionHead(torch.nn.Module):
    """
    Selects the binned RILE score (left, centre-left, centrist,
    centre-right, right) based on the embedding of a manifesto
    chunk computed by a long-input transformer model.
    """

    def __init__(self, input_dim, inner_dim=1024, outer_dim=5):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, outer_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        return self.linear2(x)


def train_epoch(epoch_n, data, batch_size,
                tokeniser, encoder, binned_regression_head,
                optimiser, loss_function):
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
        riles_predicted = binned_regression_head(cls_embeddings)
        riles_gold = torch.tensor(data.RILE[lo:hi].values).cuda()
        loss = loss_function(riles_predicted, riles_gold)
        loss.backward()
        optimiser.step()
        losses[step] = loss.item()
        predictions.extend(riles_predicted.detach().cpu().flatten().tolist())
    return losses.mean().item(), predictions


def validate_epoch(epoch_n, data, batch_size,
                   tokeniser, encoder, binned_regression_head):
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
            logits = binned_regression_head(cls_embeddings)
        riles_predicted = logits.cpu().argmax(axis=-1).flatten().tolist()
        predictions.extend(riles_predicted)
    accuracy = (
        torch.tensor(predictions) == torch.tensor(data.RILE.values)
    ).sum().item() / len(predictions)
    return accuracy, predictions


def main():
    # Preprocessing done in another script:
    # -- split translated manifestos into chunk of length less than 4096
    #    (this operation is dependent on the tokeniser/model)
    # -- compute RILEs for manifesto chunks

    seed(42)
    torch.manual_seed(42)

    chunk_file_path = '../data/chunks_for_manifestos_longformer.json'
    rile_path = '../data/RILE_by_chunk_longformer.json'

    with open(rile_path) as inp:
        rile_by_chunk = json.load(inp)

    model_name = 'allenai/longformer-base-4096'
    tokeniser = LongformerTokenizerFast.from_pretrained(model_name)

    with open(chunk_file_path) as inp:
        chunks_by_manifesto = json.load(inp)

    dataset_path = '../data/longformer_chunk_training_data.csv'
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

    # Bin RILEs
    binned_riles = all_data.RILE.map(utils.bin_rile)
    del all_data['RILE']
    all_data.insert(len(all_data.columns), 'RILE', binned_riles)
    
    print(f'Dataset size: {all_data.shape[0]}')
    n_epochs = 5
    batch_size = 4

    # Split test countries between GPUs for parallel training.
    # GPU ids are between 1 and N_GPUS.
    all_countries = set()
    for m_id in all_data.manifesto_id:
        test_country, _, _, _ = m_id.split('-')
        all_countries.add(test_country)
    all_countries = sorted(all_countries)
    countries_by_gpu_id = {}
    country_batch_size = ceil(len(all_countries) / N_GPUS)
    for i in range(1, N_GPUS + 1):
        idx = i - 1
        lo = idx * country_batch_size
        if i == N_GPUS:
            countries_by_gpu_id[i] = all_countries[lo:]
        else:
            hi = lo + country_batch_size
            countries_by_gpu_id[i] = all_countries[lo:hi]
    assert sum(
        len(v) for v in countries_by_gpu_id.values()
    ) == len(all_countries)

    current_gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
    country_batch = countries_by_gpu_id[current_gpu_id]
    print(f'Current countries: {", ".join(country_batch)}')

    # Results by country
    results_dict = {}

    for (country, 
         data_train_all, 
         data_test) in utils.split_chunk_data_loco(all_data):
        if country not in country_batch:
            continue
        indices = list(range(data_train_all.shape[0]))
        shuffle(indices)
        dev_size = len(indices) // 10
        dev_indices = indices[: dev_size]
        train_indices = indices[dev_size :]
        data_train = data_train_all.iloc[train_indices, :]
        data_dev = data_train_all.iloc[dev_indices, :]
        print(f'Splits sizes: train = {data_train.shape[0]}, '
            f'dev = {data_dev.shape[0]}, test = {data_test.shape[0]}')

        encoder = LongformerModel.from_pretrained(model_name)
        encoder.cuda()
        binned_regression_head = RILEBinnedRegressionHead(input_dim=768)
        binned_regression_head.cuda()
        optimiser = torch.optim.AdamW(
            list(encoder.parameters()) + list(binned_regression_head.parameters()), 
            lr=10**(-5))
        loss = torch.nn.CrossEntropyLoss()

        best_dev_accuracy = 0.0
        test_accuracy = 0.0
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
                binned_regression_head,
                optimiser,
                loss)
            print(f'Epoch {epoch_n+1} loss: {epoch_train_loss}')

            epoch_dev_accuracy, epoch_dev_predictions = validate_epoch(
                epoch_n,
                data_dev,
                batch_size,
                tokeniser,
                encoder,
                binned_regression_head)
            print(
                f'Epoch {epoch_n+1} dev accuracy: {epoch_dev_accuracy}')
            dev_predictions[epoch_n] = epoch_dev_predictions

            epoch_test_accuracy, epoch_test_predictions = validate_epoch(
                epoch_n,
                data_test,
                batch_size,
                tokeniser,
                encoder,
                binned_regression_head)

            if epoch_dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = epoch_dev_accuracy
                test_accuracy = epoch_test_accuracy
                test_predictions = epoch_test_predictions

        results_dict[country] = {
            'test': {
                'manifesto_ids': data_test.manifesto_id.to_list(),
                'predictions': test_predictions
            }
        }
        print(
            f'Test accuracy after {n_epochs} epochs: {test_accuracy}')
        
    with open(
        f'../results/results_longformer_binned_ovn_batch_{current_gpu_id}.json', 'w'
    ) as out:
        json.dump(results_dict, out)


if __name__ == '__main__':
    main()

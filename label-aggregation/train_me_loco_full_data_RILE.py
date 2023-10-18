# Train the classifier on the translated data without using
# label transitions.

from math import ceil
from random import shuffle, seed
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import utils

import numpy as np


class ClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, out_dim, inner_dim=1024):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        return self.linear2(x)


def train_epoch(
        epoch_n,
        epoch_train_steps,
        data,
        batch_size,
        tokeniser,
        model,
        classification_head,
        loss_function,
        optimiser,
        sbert=False
    ):
    model.train()
    epoch_losses = torch.zeros(epoch_train_steps)
    for i in tqdm(range(epoch_train_steps), desc=f'Epoch {epoch_n+1}', leave=False):
        lo = i * batch_size
        hi = lo + batch_size
        labels = torch.tensor(data['label_i'][lo:hi].to_list()).cuda()
        sentences = data['text'][lo:hi].to_list()
        tokenisation = tokeniser(sentences, padding=True, truncation=True, return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokenisation.items()}
        model_outputs = model(**model_inputs).last_hidden_state
        if sbert:
            representations = utils.mean_pooling(model_outputs, model_inputs['attention_mask'])
        else:
            # CLS embeddings
            representations = model_outputs[:, 0, :]
        logits = classification_head(representations)
        loss = loss_function(logits, labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        epoch_losses[i] = loss.item()
    return epoch_losses.mean().item()


def validate_epoch(
        epoch_n,
        n_steps,
        data,
        batch_size,
        tokeniser,
        model,
        classification_head,
        sbert=False
    ):
    model.eval()
    epoch_accuracies = torch.zeros(n_steps)
    predictions = []
    for i in tqdm(
        range(n_steps), desc=f'Epoch {epoch_n+1}, validation', leave=False
    ):
        lo = i * batch_size
        hi = lo + batch_size
        labels = torch.tensor(data['label_i'][lo:hi].to_list()).cuda()
        sentences = data['text'][lo:hi].to_list()
        tokenisation = tokeniser(
            sentences, padding=True, truncation=True, return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokenisation.items()}
        with torch.no_grad():
            model_outputs = model(**model_inputs).last_hidden_state
            if sbert:
                representations = utils.mean_pooling(
                    model_outputs, model_inputs['attention_mask'])
            else:
                # CLS embeddings
                representations = model_outputs[:, 0, :]
            logits = classification_head(representations)
        predicted_labels = torch.argmax(logits, dim=-1)
        epoch_accuracies[i] = (predicted_labels == labels).sum().item() / len(sentences)
        for j in range(predicted_labels.size(0)):
            predictions.append(predicted_labels[j].item())
    return epoch_accuracies.mean().item(), predictions


def main():
    seed(42)
    torch.manual_seed(42)

    test_run = False

    model_name = 'xlm-roberta-base'
    # model_name = 'sentence-transformers/all-mpnet-base-v2'
    sbert = 'mpnet' in model_name
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    batch_size = 32*3 # if sbert else 16*3
    # batch_size = 48*8 if sbert else 24*8
    n_epochs = 2

    all_data = pd.read_csv('../data/full_data_w_translations_cleaned.csv')
    # Remove old data because of annotation discrepansies
    all_data = all_data.loc[ all_data.year >= 2000 ]
    countries = list(all_data.country.unique())

    # In turn, predict a country from all other countries
    # Only predictions from the second epoch are saved
    predictions_by_manifesto = {}
    rile_by_manifesto = {}
    for country_idx in tqdm(range(len(countries)), desc=f'Test countries'):
        test_country = countries[country_idx]
        print(test_country)
        train_data = all_data.loc[ all_data.country != test_country ]
        test_data = all_data.loc[ all_data.country == test_country ]

        # Only leave labels that relate to the RILE score
        rile_strict = True               # Only three labels
        rile_only = rile_strict or True  # All relevant labels
        if rile_strict:
            rile_dict = utils.get_rile_categories()
            label_update = lambda l: rile_dict.get(l, 'other')
            new_train_labels = train_data.label.map(label_update)
            new_test_labels = test_data.label.map(label_update)
            del train_data['label']
            del test_data['label']
            train_data.insert(0, 'label', new_train_labels)
            test_data.insert(0, 'label', new_test_labels)
            log_file_name = 'results_full_data_me_loco_RILE_strict.log'
        elif rile_only:
            rile_dict = utils.get_rile_categories()
            label_update = lambda l: l if l in rile_dict else 'other'
            new_train_labels = train_data.label.map(label_update)
            new_test_labels = test_data.label.map(label_update)
            del train_data['label']
            del test_data['label']
            train_data.insert(0, 'label', new_train_labels)
            test_data.insert(0, 'label', new_test_labels)
            log_file_name = 'results_full_data_me_loco_RILE_only.log'
        else:
            log_file_name = 'results_full_data_me_loco.log'
        train_labels = set(train_data.label.unique())

        if not test_run:
            print(f'Writing to {log_file_name}')
            with open(log_file_name, 'a') as out:
                print(f'Test country: {test_country}', file=out)
                print(f'{train_data.shape=}', file=out)
                print(f'{test_data.shape=}', file=out)
        print(f'Test country: {test_country}')
        print(f'{train_data.shape=}')
        print(f'{test_data.shape=}')
        print()

        # We cannot predict previously unsees labels, so we convert them to 'other'
        # Not relevant for RILE only and especially for strict RILE
        seen_in_training = lambda l: l if l in train_labels else 'other'
        new_labels = test_data.label.map(seen_in_training)
        del test_data['label']
        test_data.insert(0, 'label', new_labels)
        test_labels = set(test_data.label.unique())
        assert len(test_labels - train_labels) == 0, \
            f'Test labels not seen in training: {sorted(test_labels - train_labels)}'

        # Recode labels with integers for one-hot encoding and predicting
        label_to_idx = {label: idx for idx, label in enumerate(train_labels)}
        label_to_idx_fun = lambda l: label_to_idx[l]
        train_data.insert(0, 'label_i', train_data.label.map(label_to_idx_fun))
        test_data.insert(0, 'label_i', test_data.label.map(label_to_idx_fun))
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # For computing the RILE
        test_data.insert(
            0, 
            'manifesto_id', 
            test_data.apply(utils.get_manifesto_id, axis=1))

        # Separate the dev set
        idx_arr = [i for i in range(train_data.shape[0])]
        shuffle(idx_arr)
        dev_data_boundary = len(idx_arr) // 10
        dev_idx = idx_arr[: dev_data_boundary]
        train_idx = idx_arr[dev_data_boundary :]
        dev_data = train_data.iloc[dev_idx, :]
        train_data = train_data.iloc[train_idx, :]

        if test_run:        
            print('!!! Test run !!!')
            train_data = train_data.iloc[:10000,]
            dev_data = dev_data.iloc[:1000,]
            test_data = test_data.iloc[:1000,]

        model = AutoModel.from_pretrained(model_name)
        model.cuda()

        n_classes = len(label_to_idx)
        classification_head = ClassificationHead(
            input_dim=768,
            out_dim=n_classes)
        classification_head.cuda()

        optimiser = torch.optim.AdamW(
            list(model.parameters()) + list(classification_head.parameters()),
            lr=0.00001)
        
        model = torch.nn.DataParallel(model)
        classification_head = torch.nn.DataParallel(classification_head)

        loss_function = torch.nn.CrossEntropyLoss()
        epoch_train_steps = ceil(train_data.shape[0] / batch_size)
        epoch_dev_steps = ceil(dev_data.shape[0] / batch_size)
        epoch_test_steps = ceil(test_data.shape[0] / batch_size)
        for epoch_n in range(n_epochs):
            epoch_train_loss = train_epoch(
                epoch_n,
                epoch_train_steps,
                train_data,
                batch_size,
                tokeniser,
                model,
                classification_head,
                loss_function,
                optimiser,
                sbert=sbert)
            epoch_dev_accuracy, _ = validate_epoch(
                epoch_n,
                epoch_dev_steps,
                dev_data,
                batch_size,
                tokeniser,
                model,
                classification_head,
                sbert=sbert)
            epoch_test_accuracy, epoch_test_predictions = validate_epoch(
                epoch_n,
                epoch_test_steps,
                test_data,
                batch_size,
                tokeniser,
                model,
                classification_head,
                sbert=sbert)
            print(f'{test_country}, {epoch_n+1}, {epoch_train_loss=}, {epoch_dev_accuracy=}, {epoch_test_accuracy=}')
            if not test_run:
                with open(log_file_name, 'a') as out:
                    print(f'{test_country}, {epoch_n+1}, {epoch_train_loss=}, {epoch_dev_accuracy=}, {epoch_test_accuracy=}', file=out)

            test_data_copy = test_data.copy()
            del test_data_copy['label_i']
            test_data_copy.insert(0, 'label_i', epoch_test_predictions)
            n_manifestos = len(test_data.manifesto_id.unique())
            gold_riles = np.zeros(n_manifestos)
            predicted_riles = np.zeros(n_manifestos)
            for i, manifesto_id in enumerate(test_data.manifesto_id.unique()):
                gold_labels = list(test_data.loc[test_data.manifesto_id == manifesto_id].label)
                predicted_labels = [
                    idx_to_label[el] for el in 
                    test_data_copy.loc[test_data_copy.manifesto_id == manifesto_id].label_i
                ]
                if rile_strict:
                    gold_riles[i] = utils.compute_rile_simple(gold_labels)
                    predicted_riles[i] = utils.compute_rile_simple(predicted_labels)
                else:
                    gold_riles[i] = utils.compute_rile_from_list(gold_labels)
                    predicted_riles[i] = utils.compute_rile_from_list(predicted_labels)
                if epoch_n == 1:
                    predictions_by_manifesto[manifesto_id] = predicted_labels
                    rile_by_manifesto[manifesto_id] = {
                        'predicted': predicted_riles[i],
                        'gold': gold_riles[i]
                    }
            print(
                'RILE correlation coefficient:',
                np.corrcoef(gold_riles, predicted_riles)[1, 0])
            if not test_run:
                with open(log_file_name, 'a') as out:
                    print(
                        'RILE correlation coefficient:',
                        np.corrcoef(gold_riles, predicted_riles)[1, 0],
                        file=out)
                if epoch_n == 1:
                    with open(log_file_name.replace('.log', '.json'), 'w') as out:
                        json.dump(rile_by_manifesto, out, indent=2)
                    with open(log_file_name.replace('.log', '_predictions.json'), 'w') as out:
                        json.dump(predictions_by_manifesto, out, indent=2)


if __name__ == '__main__':
    main()

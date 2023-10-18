import json
import glob
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


CODE2COUNTRY = {
    'at': 'Austria',
    'be': 'Belgium',
    'ch': 'Switzerland',
    'de': 'Germany',
    'dk': 'Denmark',
    'es': 'Spain',
    'fi': 'Finland',
    'it': 'Italy',
    'nl': 'Netherlands',
    'se': 'Sweden'
}
COUNTRY2CODE = {v: k for k, v in CODE2COUNTRY.items()}

RILE_MAPPING_STR = """{
    "right": {
        "Civic Mindedness: Positive": "606",
        "Constitutionalism: Positive": "203",
        "Economic Incentives": "402",
        "Economic Orthodoxy": "414",
        "Free Market Economy": "401",
        "Freedom and Human Rights": "201",
        "Law and Order: Positive": ["605", "605.1"],
        "Military: Positive": "104",
        "National Way of Life: Positive": "601",
        "Political Authority": "305",
        "Protectionism: Negative": "407",
        "Traditional Morality: Positive": "603",
        "Welfare State Limitation": "505"
    },
    "left": {
        "Anti-imperialism": "103",
        "Controlled Economy": "412",
        "Democracy": "202",
        "Economic Planning": "404",
        "Education Expansion": "506",
        "Internationalism: Positive": "107" ,
        "Labour Groups: Positive": "701",
        "Market Regulation": "403",
        "Military: Negative": "105",
        "Nationalisation": "413",
        "Peace": "106",
        "Protectionism: Positive": "406",
        "Welfare State Expansion": "504"
    }
}
"""

GAL_TAN_MAPPING_STR = """{
    "authoritarian": {
        "Political Authority": ["305", "305.1"],
        "National Way of Life: Positive": ["601", "601.1"],
        "Traditional Morality: Positive": "603",
        "Law and Order: Positive": ["605", "605.1"],
        "Multiculturalism: Negative": ["608", "608.1"],
        "Social harmony": ["606", "606.1"]
    },
    "libertarian": {
        "Environmental protection": "501",
        "National Way of Life: Negative": ["602", "602.1"],
        "Traditional Morality: Negative": "604",
        "Culture: Positive": "502",
        "Multiculturalism: Positive": ["607", "607.1"],
        "Anti-Growth Economy: Positive": ["416", "416.1"],
        "Underprivileged Minority Groups": "705",
        "Non-economic Demographic Groups": "706",
        "Freedom and Human Rights": ["201", "201.1", "201.2"],
        "Democracy": ["202", "202.1"]
    }
}
"""


def get_rile_categories():
    result = {}
    mapping = json.loads(RILE_MAPPING_STR)
    for k in mapping:
        for v in mapping[k].values():
            if type(v) == list:
                for el in v:
                    result[el] = k
            else:
                result[v] = k
    return result


def get_gal_tan_categories():
    result = {}
    mapping = json.loads(GAL_TAN_MAPPING_STR)
    for k in mapping:
        for v in mapping[k].values():
            if type(v) == list:
                for el in v:
                    result[el] = k
            else:
                result[v] = k
    return result


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_one_hot(label_i, n_classes):
    """
    Get one-hot encoding of a single label
    for one-by-one predictions.
    """
    return torch.nn.functional.one_hot(
        torch.tensor([label_i]), 
        num_classes=n_classes
    ).float().cuda()


def normalise_label(l):
    l = str(l)
    l = l.strip()
    # Not all manifestos distinguish between no-label and heading (H),
    # so we collapse them into 'other'.
    return l[:-2] if l.endswith('.0') else (
        'other' if l in {'H', '000'} else l
    )


def load_manifesto_w_translations(path):
    """
    The files with translations we are using now already
    have all the metadata + left context in the original
    language.
    """
    return pd.read_csv(path, dtype='object').fillna('other')


def load_data_for_countries_w_translations(country_codes, show_progress_bar=True):
    data_arr = []
    for country in tqdm(country_codes, desc='Reading the data', leave=False, disable=(not show_progress_bar)):
        files = glob.glob(f'../data_w_translations/{country}-*.csv')
        for f in tqdm(files, leave=False, disable=(not show_progress_bar)):
            df = load_manifesto_w_translations(f)
            data_arr.append(df)
    return pd.concat(data_arr)


def compute_confusion_matrix(real_labels, predicted_labels, label_to_idx):
    idx_to_label = { v: k for k, v in label_to_idx.items() }
    labels = sorted(label_to_idx)
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for l_true, l_pred in zip(real_labels, predicted_labels):
        l_true_str = idx_to_label[l_true]
        l_pred_str = idx_to_label[l_pred.item()]
        cm.loc[l_true_str, l_pred_str] += 1
    return cm


def compute_rile_simple(labels):
    """
    Compute the RILE score from a list of labels containing 'right', 'left', and anything else.
    """
    R = sum(1 for l in labels if l == 'right')
    L = sum(1 for l in labels if l == 'left')
    return (R - L) / len(labels)


def compute_rile_from_list(labels, rile_dict=None):
    """
    Compute the RILE score from a list of CMP labels.
    """
    if rile_dict is None:
        rile_dict = get_rile_categories()
    labels = [rile_dict.get(label, 'neutral') for label in labels]
    return compute_rile_simple(labels)


def compute_rile(manifesto_df, rile_dict=None):
    """
    Compute the RILE score from a CMP data-frame.
    """
    if rile_dict is None:
        rile_dict = get_rile_categories()
    labels = [rile_dict.get(label, 'neutral') for label in manifesto_df.label]
    return compute_rile_simple(labels)


def compute_gal_tan_simple(labels):
    """
    Compute the GAL–TAN score from a list of labels containing 'libertarian', 
    'authoritarian', and anything else.
    """
    A = sum(1 for l in labels if l == 'authoritarian')
    L = sum(1 for l in labels if l == 'libertarian')
    return (A - L) / len(labels)


def compute_gal_tan_from_list(labels, gal_tan_dict=None):
    """
    Compute the GAL–TAN score from a list of CMP labels.
    """
    if gal_tan_dict is None:
        gal_tan_dict = get_gal_tan_categories()
    labels = [gal_tan_dict.get(label, 'neutral') for label in labels]
    return compute_gal_tan_simple(labels)


def compute_gal_tan(manifesto_df, gal_tan_dict=None):
    """
    Compute the GAL–TAN score from a CMP data-frame.
    """
    if gal_tan_dict is None:
        gal_tan_dict = get_gal_tan_categories()
    labels = [gal_tan_dict.get(label, 'neutral') for label in manifesto_df.label]
    return compute_gal_tan_simple(labels)


def get_manifesto_id(row):
    month = str(row.month).rjust(2, '0')
    return f'{row.country}-{row.party}-{row.year}-{month}'


def bin_rile(rile_score):
    assert rile_score >= -1.0 and rile_score <= 1.0, f'RILE score outside the allowed range of [-1, 1]: {rile_score}'
    if rile_score < -0.6:
        return 0  # hard left
    elif rile_score < -0.2:
        return 1  # centre left
    elif rile_score < .2:
        return 2  # centrist
    elif rile_score < .6:
        return 3  # centre right
    else:
        return 4  # hard right
    

def split_chunk_data_old_vs_new(all_data: pd.DataFrame):
    after_2019 = []
    for m_id in all_data.manifesto_id:
        _, _, year, _ = m_id.split('-')
        after_2019.append(int(year) >= 2019)
    train_data = all_data.loc[np.logical_not(after_2019)]
    test_data = all_data.loc[after_2019]
    return train_data, test_data


def split_chunk_data_loco(all_data: pd.DataFrame):
    countries = []
    for m_id in all_data.manifesto_id:
        test_country, _, _, _ = m_id.split('-')
        countries.append(test_country)
    all_countries = set(countries)
    countries = np.array(countries)
    for test_country in sorted(all_countries):
        train_data = all_data.loc[countries != test_country]
        test_data = all_data.loc[countries == test_country]
        yield test_country, train_data, test_data
    
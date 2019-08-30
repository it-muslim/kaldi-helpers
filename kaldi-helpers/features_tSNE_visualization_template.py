"""Visualize features

This is a template for features t-SNE visualization colored by phoneme.
In fact, t-SNE can be replaced by any other dimension reduction method.
"""
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from kaldi_io import phone_int2symb, read_ali, read_feats

# Set paths and commands
project_root = "<project root path>"  # Like /abs/path/to/kaldi-for-dummies"
data = os.path.join(project_root, "data/train")
model_dir = os.path.join(project_root, "exp/mono")
feats_command = f"apply-cmvn --utt2spk=ark:{data}/utt2spk scp:{data}/cmvn.scp scp:{data}/feats.scp ark:- | add-deltas ark:- ark,t:-"

# Read alis, feats, and phone mapping(int->symb)
phones_mapping = phone_int2symb(os.path.join(model_dir, "phones.txt"))
feats = read_feats(feats_command)
alis = read_ali(model_dir)

# Make sure feats and alis have same utterances in exactly the same order
feat_utterances = set(feats.keys())
ali_utterances = set(alis.keys())
if feat_utterances != ali_utterances:
    for k in feat_utterances.difference(ali_utterances):
        print(f"WARNING: There is no alignment for {k}, so removing it")
        del feats[k]
    for k in ali_utterances.difference(feat_utterances):
        print(f"WARNING: There are no feats for {k}, so removing it")
        del alis[k]
    assert set(feats.keys()) == set(alis.keys())
    feats = OrderedDict(feats)
alis = OrderedDict(alis)
assert feats.keys() == alis.keys()

# Join values of all utterances into one matrix/array
X = np.vstack(tuple(feats.values()))
phones = np.concatenate(tuple(alis.values()))
utterances = []
for utter in alis.keys():
    utterances = utterances + [utter] * len(alis[utter])
assert X.shape[0] == len(phones)
assert len(utterances) == len(phones)
# Category needs less memory and works faster
phones = pd.Series(phones, dtype="category").cat.rename_categories(phones_mapping)
utterances = pd.Series(utterances, dtype="category")

# Restrict number of rows
indexes = np.random.choice(np.arange(0, X.shape[0]), 2500, replace=False)

# Compute t-SNE
X_transformed = TSNE(n_components=2, init="pca", perplexity=100).fit_transform(
    X[indexes, :]
)

# Plot features colored by phoneme
sns.scatterplot(X_transformed[:, 0], X_transformed[:, 1], hue=phones)
plt.show()

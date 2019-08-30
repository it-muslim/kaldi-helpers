"""
Helper functions to compute GOP (Goodness of Pronunciation).
Basic concept (for each utterance):
1. Align data using lexicon and a model (forced alignment)
Output: a vector of length M, where M is number of frames.
Values are phonemes according to the alignment. I.e. some simulation of "real" values for a frame.
2. For the same utterance predict phonemes using the model,
but without the lexicon.
Output: MxN matrix, where M is number of frames, N is the number of phonemes,
values are probability for the corresponding phoneme to be at the frame with respect to the model.
3. If the model is good enough then for each frame alignment value must be the most probable one.
NOTE: It is assumed that all necessary commands exist (i.e are added to PATH).
"""

import os
import glob
from typing import Optional

import numpy as np
import pandas as pd

from kaldi_io import phone_symb2int, pdf2phone, read_ali, read_feats
from utils import softmax


def compute_gmm_probs(
    model_dir: str, feats: str, ali_rspec: Optional[str] = None
) -> pd.DataFrame:
    """Compute GOP probs for a GMM based model

    Computes probs for GOP using a GMM model. The algorithm is (for each utterance):
    1. Estimate 'real' phonemes for each frame by alignment
        (by `ali-to-phones --per-frame=true <model> <ali_rspec> ark,t:-`):
        Output: Vector of length M, M - number of frames
    2. Compute log-likelihoods of pdf_ids for each frame
        (by `gmm-compute-likes {model} {feats} ark,t:-`)
        Output: Matrix MxN, M - number of frames, N - number of pdf_ids
    3. Convert log-likelihoods to probabilities by softmax transformation:
        probs[i,:] = softmax(likes[i,:])
        Output: Matrix MxN, M - number of frames, N - number of pdf_ids.
    4. Map pdf_id to phonemes by `show-transitions`
        (see kaldi_io.pdf2phone for details)
    5. Sum pdf_id-probabilities by phoneme:
        probs_new[i,j] = sum(probs[i, <all pdf_ids of phoneme j>])
        Output: Matrix MxK, K - number of phonemes
    6. For each frame get:
        - predicted_phone = argmax(probs_new[i,:]),
        - predicted_phone_prob = max(probs_new[i,:]),
        - real_phone = alignment_phoneme[i]
        - real_phone_prob = probs_new[i,alignment_phoneme[i]],
        This is one row of the output DataFrame

    Parameters
    ----------
    model_dir : str
        A dir to a gmm model. Supposed to be standard kaldi model. For example, 'exp/mono_mfcc'
    feats: str
        A rspec for feats to provide for gmm-compute-likes. For example,
        feats = f'"ark,s,cs:apply-cmvn --utt2spk=ark:{data}/utt2spk scp:{data}/cmvn.scp scp:{data}/feats.scp ark:- | add-deltas ark:- ark:- |"'
    ali_rspec: str, optional
        Standard kaldi rspec for an alignment file. For example, 'ark:"gunzip -c exp/mono_mfcc/ali.1.gz|"
        By default all ali.*.gz files from model_dir are taken.

    Returns
    -------
    probs: pd.DataFrame
        A pandas DataFrame of following structure:
          - utterance - name of utterance
          - frame - number of the frame
          - predicted_phone - a phoneme having maximum probability for the frame (i.e. argmax for probability)
          - predicted_phone_prob - maximum phoneme-probability for the corresponding frame
          - real_phone - a phoneme expected by alignment
          - real_phone_prob - probability of the phoneme expected by the alignment
    """
    model = os.path.join(model_dir, "final.mdl")
    phones_file = os.path.join(model_dir, "phones.txt")

    # Get mappings between phoneme int code and symbol
    symb2int = phone_symb2int(phones_file)

    # Get mapping for pdf id -> phoneme symbol
    pdf2symb = pdf2phone(model_dir)
    pdf2int = {k: symb2int[v] for k, v in pdf2symb.items()}

    # Get alignments
    alis = read_ali(model, ali_rspec)

    # Compute gmm probs for each pdf id
    gmm_likes_command = f"gmm-compute-likes {model} {feats} ark,t:-"
    gmm_likes = read_feats(gmm_likes_command)

    # Calculate GOP for each utterance
    probs_summary = {}
    for utterance, likes in gmm_likes.items():
        try:
            ali = alis[utterance]
        except KeyError:
            print(f"Could not find alignment for {utterance}")
            continue

        # Convert likes to probabilities by softmax transformation
        prob = softmax(likes)

        # Switch from pdf_id to phoneme: sum probs by phoneme
        prob = pd.DataFrame(data=prob, columns=pdf2int.values())
        prob = prob.groupby(by=prob.columns, axis=1).sum()

        # Get probs summary:
        #   max probability phoneme (i.e. predicted phoneme)
        #  VS.
        #   alignment phoneme (i.e. real phoneme)
        probs_summary[utterance] = pd.DataFrame(
            {
                "predicted_phone": prob.idxmax(axis=1).astype(np.uint16),
                "predicted_phone_prob": prob.max(axis=1).astype(np.float32),
                "real_phone": ali.astype(np.uint16),
                "real_phone_prob": prob.lookup(np.arange(prob.shape[0]), ali).astype(
                    np.float32
                ),
            }
        )
    # del gmm_likes

    # Convert dict of DataFrames to one DataFrame
    return pd.concat(probs_summary.values(), keys=probs_summary.keys())


def summarize_probs(
    probs: pd.DataFrame, by: Optional[str] = None, phones_file: Optional[str] = None
) -> pd.DataFrame:
    """Summarize probs tables like output of `compute_gmm_probs`

    Calculates phone level errors metrics. See output description for output
    details. Values can be calculated grouped by utterance or phone. Use `by`
    parameter for that.

    Parameters
    ----------
    probs : pd.DataFrame
        Table of predicted probs for utterances. See `compute_gmm_probs` for
        the table structure.
    by : Optional[str], optional
        Either None, 'phone', or 'utterance'. Provides the mode of
        summarization. By default, all metrics are calculated overall frames
        of all utterances. 'phone' means that metrics will be calculated per
        phoneme, i.e. for each `real_phone` separately. Similarly, 'utterance'
        means that metrics will be calculated for each
    phones_file : str, optional
        A path to a phones.txt file. For example, 'exp/mono_mfcc/phones.txt'

    Returns
    -------
    pd.DataFrame
        A summary table of following structure:
         - median r/p ratio - median value of `real_phone_prob / predicted_phone_prob` ratio
         - mean r/p ratio - mean value of `real_phone_prob / predicted_phone_prob` ratio
         - median_prob - median value of `real_phone_prob`
         - mean_prob - mean value of `real_phone_prob`
         - PER - Phone Error Rate - ratio of incorrectly predicted frames, i.e. `correct_number / total`
         - correct_number - number of correctly predicted frames
         - total - total number of frames
        For `phone` mode it also symbolic codes of phones are also added

    Raises
    ------
    ValueError
        If `by` value is not in (None, 'phone', 'phoneme', 'utt', 'utter', 'utterance')
    """
    # Set grouping vector to group values by it
    if by is None:
        mode = "overall"
        grouping_vector = [True] * probs.shape[0]
    elif by in ("phone", "phoneme"):
        mode = "phone"
        grouping_vector = probs.real_phone
    elif by in ("utt", "utter", "utterance"):
        mode = "utterance"
        grouping_vector = probs.index.get_level_values(0)
    else:
        raise ValueError("Unknown value of 'by' parameter")

    # Compute grouped summaries
    summary_table = pd.concat(
        {
            "median r/p ratio": (probs.real_phone_prob / probs.predicted_phone_prob)
            .groupby(grouping_vector)
            .median()
            .astype(np.float32),
            "mean r/p ratio": (probs.real_phone_prob / probs.predicted_phone_prob)
            .groupby(grouping_vector)
            .mean()
            .astype(np.float32),
            "median_prob": probs.real_phone_prob.groupby(grouping_vector)
            .median()
            .astype(np.float32),
            "mean_prob": probs.real_phone_prob.groupby(grouping_vector)
            .mean()
            .astype(np.float32),
            "correct_number": (probs.real_phone_prob == probs.predicted_phone_prob)
            .astype(int)
            .groupby(grouping_vector)
            .sum(),
            "total": pd.Series(grouping_vector).value_counts(sort=False),
        },
        axis=1,
        sort=True,
    )
    summary_table["PER"] = 100 * (
        1 - summary_table["correct_number"] / summary_table["total"]
    )
    # Rearrange columns
    summary_table = summary_table[
        [
            "median r/p ratio",
            "mean r/p ratio",
            "median_prob",
            "mean_prob",
            "PER",
            "correct_number",
            "total",
        ]
    ]

    # Add mappings between phoneme int code and symbol
    if (mode == "phone") and (phones_file is not None):
        symb2int = phone_symb2int(phones_file)
        summary_table = pd.concat(
            (
                pd.Series(
                    list(symb2int.keys()),
                    index=list(symb2int.values()),
                    name="phone_symbol",
                ),
                summary_table,
            ),
            axis=1,
            sort=True,
        )

    # Remove dummy group for overall mode
    if mode == "overall":
        summary_table.reset_index(drop=True, inplace=True)
    else:
        summary_table.index.name = mode
        summary_table.reset_index(inplace=True)

    # Sort values: from best to worst
    summary_table.sort_values("PER", ascending=False, inplace=True)

    return summary_table

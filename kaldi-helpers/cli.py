import os
from typing import Optional

import fire

from phone_probs import compute_gmm_probs, summarize_probs


class CliCommands:
    """Dummy class to group cli commands"""

    def analize_phone_errors(
        self,
        model_dir: str,
        feats: str,
        outputdir: Optional[str] = None,
        modeltype: str = "gmm",
    ) -> None:
        """Analyze phone level errors

        Analyzes phone level errors and prints output to 3 csv files:
        - per_overall.csv - statistics overall data
        - per_by_phone.csv - statistics grouped by phoneme
        - per_by_utterance.csv - statistics grouped by utterance

        Parameters
        ----------
        model_dir : str
            A dir to the model. Supposed to be standard kaldi model. For example, 'exp/mono_mfcc'
        feats : str
            rspec for feats
        outputdir : Optional[str], optional
            A dir where to put outputs
        modeltype : str, optional
            Model type either 'gmm' or 'nn'. By default 'gmm' is assumed
        """
        # If model_dir is a *.mdl file, then take dirname
        if os.path.isfile(model_dir):
            model_dir = os.path.dirname(model_dir)
        # For some reason quote marks disappears when values is sent by cli
        # So, we add quote marks manually and strip for just in case.
        feats = '"' + feats.strip("'\"") + '"'
        # By default put outputs to model_dir
        if outputdir is None:
            outputdir = model_dir
        # Check model type
        if modeltype == "gmm":
            probs = compute_gmm_probs(model_dir, feats)
        elif modeltype == "nn":
            raise ValueError("Compute probs for NN models is not implemented yet.")
        else:
            raise ValueError("Unknown model type.")

        # Get overall summary
        summary = summarize_probs(probs)
        summary.to_csv(os.path.join(outputdir, "per_overall.csv"), index=False)

        # Get per phone summary
        summary = summarize_probs(
            probs, "phone", phones_file=os.path.join(model_dir, "phones.txt")
        )
        summary.to_csv(os.path.join(outputdir, "per_by_phone.csv"), index=False)

        # Get per utterance summary
        summary = summarize_probs(probs, "utterance")
        summary.to_csv(os.path.join(outputdir, "per_by_utterance.csv"), index=False)

        return


if __name__ == "__main__":
    fire.Fire(CliCommands)

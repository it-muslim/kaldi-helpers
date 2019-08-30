# HQA Kaldi Helpers

These are helper scrips for working with kaldi projects.

## Available functions and examples

:warning: **NOTE:** It is supposed that kaldi functions are added to the PATH. I.e. do `source path.sh` in one of your projects before run a script.

### `pdf2phone`
Map pdf_id(s) to actual phones
Uses kaldi's show-transition command, which gives output like:
```
Transition-state 134: phone = ws hmm-state = 0 pdf = 43
 Transition-id = 283 p = 0.916493 count of pdf = 4323 [self-loop]
 Transition-id = 284 p = 0.0835068 count of pdf = 4323 [0 -> 1]
Transition-state 135: phone = ws hmm-state = 1 pdf = 124
 Transition-id = 285 p = 0.896085 count of pdf = 3474 [self-loop]
 Transition-id = 286 p = 0.103915 count of pdf = 3474 [1 -> 2]
Transition-state 136: phone = ws hmm-state = 2 pdf = 109
 Transition-id = 287 p = 0.876454 count of pdf = 2922 [self-loop]
 Transition-id = 288 p = 0.123546 count of pdf = 2922 [2 -> 3]
```
All lines starting with "Transition-state" are considered as containing
mapping for pdf_id and phone symbol.

**EXAMPLE**:
```python
>>> pdf2phone("exp/mono_mfcc")
{43:"ws", 124:"ws", 109:"ws", 50: "h", ...}
```

### `phone_symb2int`
Map phoneme symbols to phoneme integer codes from phones.txt file

**EXAMPLE**:
```python
>>> phone_symb2int(exp/mono_mfcc/phones.txt)
{"a": 10, "ws": 5, ...}
```

### `phone_int2symb`
Is opposite to phone_symb2int

### `read_feats`
Reading from stdout, import feats(or feats-like) data as a numpy array

As feats are generated "on-fly" in kaldi, there is no a feats file
(except most simple cases like raw mfcc, plp or fbank).  So, that is why
we take feats as a command rather that a file path. Can be applied to
other commands (like gmm-compute-likes) generating an output in same
format as feats, i.e:
```
utterance_id_1  [
  70.31843 -2.872698 -0.06561285 22.71824 -15.57525 ...
  78.39457 -1.907646 -1.593253 23.57921 -14.74229 ...
  ...
  57.27236 -16.17824 -15.33368 -5.945696 0.04276848 ... -0.5812851 ]
utterance_id_2  [
  64.00951 -8.952017 4.134113 33.16264 11.09073 ...
  ...
```
**EXAMPLE**:
```python
>>> read_feats("copy-feats scp:data/test/feats.scp ark,t:-")
# The output is like:
{
"utterance_id_1": numpy.array([
        [70.31843, -2.872698, -0.06561285, 22.71824, -15.57525, ...],
        [78.39457, -1.907646, -1.593253, 23.57921, -14.74229, ...],
        ...
    ]),
"utterance_id_2": numpy.array([
        [64.00951, -8.952017, 4.134113, 33.16264, 11.09073, ...],
        ...
    ]),
...
}
>>> read_reats("gmm-compute-likes exp/mono_mfcc/final.mdl \"ark,s,cs:apply-cmvn --utt2spk=ark:train/utt2spk scp:train/cmvn.scp scp:train/feats.scp ark:- | add-deltas ark:- ark:- |\" ark,t:-")
# The output is a similar dictionary
```

### `read_ali`
Reading from stdout, import alignments as a numpy array

**EXAMPLE**:
```python
>>> read_ali("exp/mono_mfcc") # the same as read_ali("exp/mono_mfcc/final.mdl")
# Will collect alignments from all ali.*.gz files in "exp/mono_mfcc"
# The output is like:
{
"utterance_id_1": numpy.array([1, 1, 1, 16, 16, 16, 32, 32, ...]),
"utterance_id_2": numpy.array([1, 1, 1, 7, 7, 7, 2, 2, 2, ...]),
...
}

# Also works with rspec
>>> read_ali("exp/mono_mfcc", 'ark:"gunzip -c exp/mono_mfcc/ali.1.gz|"')
```

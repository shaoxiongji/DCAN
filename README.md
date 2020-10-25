# DCAN

Dilated Convolutional Attention Network (DCAN), integrating dilated convolutions, residual connections, and label attention, for medical code assignment. It adopts dilated convolutions to capture complex medical patterns with a receptive field which increases exponentially with dilation size.

## Data
Download MIMIC-III dataset from [physionet](https://mimic.physionet.org).

Organize your data using the following structure

```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv
```


`ICD9_descriptions` is avaiable [in this repo](https://github.com/jamesmullenbach/caml-mimic/blob/master/mimicdata/ICD9_descriptions), and 
`*_hadm_ids.csv` are avaiable [here](https://github.com/jamesmullenbach/caml-mimic/tree/master/mimicdata/mimic3).
`MIMIC_RAW_DSUMS` is available [here](https://physionet.org/works/ICD9CodingofDischargeSummaries/), while the rest file for MIMIC2 can be generated with their code. 
If you use Python3 `consctruct_datasest.py` in `ICD9_Coding_of_Discharge_Summaries` to create data files, remember to convert dict object to list (line 82&83) and use `dict.items()` instead of `dict.iteritems()`.
Assign the directories of MIMIC data using `MIMIC_3_DIR`.

## Run
``python3 main.py``

Configs available at `options.py`.

Requirements:
- python 3.7
- pytorch 1.5.0

## Citation
```
@inproceedings{ji2020dilated,
  title={Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text},
  author={Ji, Shaoxiong and Cambria, Erik and Marttinen, Pekka},
  booktitle={3rd Clinical Natural Language Processing Workshop at EMNLP},
  year={2020}
}
```

## References
- https://github.com/jamesmullenbach/caml-mimic
- https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
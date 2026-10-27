# DOREMI: Optimizing Long Tail Predictions in Document-Level Relation Extraction (v1.0)

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

<!-- For demos and web applications, also show this line in the user interface:
"Research prototype of the HEREDITARY project. Not a medical device and not for clinical use. Do not enter personal or patient data." -->

## 1. Overview

| Field | Value |
|---|---|
| Name | DOREMI: Optimizing Long Tail Predictions in Document-Level Relation Extraction |
| Version | 1.0 |
| Type | Software |
| Owner partner and contact | UNIPD, Laura Menotti, laura.menotti@unipd.it |
| License | MIT License (see LICENSE file) |
| Repository and DOI | https://github.com/mntlra/DOREMI/tree/main |
| Release date | 09/01/2026 |
| Related task or deliverable | T4.6|
| How to cite | Laura Menotti, Stefano Marchesin, and Gianmaria Silvello. 2026. DOREMI: Optimizing Long Tail Predictions in Document-Level Relation Extraction. Knowledge-Based Systems, 337, 115359 (2026). https://doi.org/10.1016/j.knosys.2026.115359 |

## 2. Intended use

**Primary intended use.** Research use

**Intended users.** Researchers, developers.

**Out-of-scope uses.** This release must not be used for
- clinical diagnosis, prognosis or treatment of individual patients;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with other data sources;
- populations, devices or data types not described in Section 3.

## 3. Data provenance

| Dataset | Source Publication | Type (public, synthetic, pseudonymized, anonymized) | Population (size, age, sex, country, condition) | Ethics approval or access conditions |
|---|---|---|---|---|
| DocRED | Yao et al (2019). https://aclanthology.org/P19-1074/ https://github.com/thunlp/DocRED | Public | Dataset constructed from Wikipedia and Wikidata with annotated named entities and relations | Publicly available |
| ReDocRED | Tan et al (2022). https://aclanthology.org/2022.emnlp-main.580/ https://github.com/tonytan48/re-docred | Public | Dataset constructed from Wikipedia and Wikidata with annotated named entities and relations | Publicly available |
| UGDRE | Sun et al (2023). https://aclanthology.org/2023.acl-long.889/ https://https://github.com/QiSun123/UGDRE | Public | Dataset constructed from Wikipedia and Wikidata with annotated named entities and relations | Publicly available |

This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys.

## 4. Method

- **Algorithm or architecture.** This repository contains the implementation of DOcument-level Relation Extraction optiMizing the long taIl (DOREMI), an active learning framework enhancing the representation of underrepresented relations through minimal yet targeted manual annotation.
- **Training setting.** Centralized (Training parameters reported in files in ```data/scripts/```)
- **Privacy techniques.** None (used datasets are public)

## 5. Performance

- **Evaluation data.** Docred and ReDocRED dev and test split (publicly available).
- **Metrics and results.** Precision, Recall, and F1. Results are available in the reference publication.
- **Results by subgroup.** No release-wide demographic subgroup results are reported.

## 6. Known limitations and risks

- **Generalizability.** DOREMI can be trained on any DocRE dataset.
- **Known biases and confounders.** The leading DocRE models exploited for empirical analysis (ATLOP and DREEAM) are transformer-based models, relying on pre-trained language models such as BERT and RoBERTa. We recognize that pretrained language models may include hidden biases from their training data, potentially embedding subtle human prejudices. While they perform well in detecting relations, special caution is needed for sensitive relation types or entities, where such biases may become more evident.
- **Failure modes.** N/A
- **Privacy risks.** N/A

## 7. Responsible use and reporting

- **Terms of use.** Released under the MIT license. The software is intended for research use only and has not been validated for clinical applications.
- **Contact for misuse or vulnerability reports.** Laura Menotti, laura.menotti@unipd.it

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.

# Attribution

`src/sepsis_utility.py` contains `compute_prediction_utility`, taken from the
official PhysioNet/CinC 2019 Challenge evaluation script
(`evaluate_sepsis_score.py`, physionetchallenges/evaluation-2019, BSD-licensed).
It is reused unchanged so this project reports the exact utility metric used by
the Challenge organizers and every published entry. The cohort-normalization
wrapper `normalized_utility` reimplements the same normalization the official
script performs across a directory of files, but in memory.

## Data

The model is trained on, and `src/static/demo_patients.json` contains seven
held-out patients from, the PhysioNet/CinC 2019 Challenge training data,
released under the Creative Commons Attribution 4.0 International License.
The patient files are redistributed unmodified apart from dropping empty
(NaN) values.

Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B.,
Sharma, A., Nemati, S., & Clifford, G. D. (2019). Early Prediction of Sepsis
from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019
(version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857

Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical
Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care
Medicine 48(2):210-217 (2020). https://doi.org/10.1097/CCM.0000000000004145

Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a
new research resource for complex physiologic signals. Circulation 101(23):
e215-e220 (2000).

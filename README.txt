### Requirements
The code was tested in the following setting:
1. Ubuntu 20.04 LTS
2. conda 4.8.4 (Python 3.7.6)
3. numpy 1.18.1
4. scikit-learn 0.22.1
5. pandas 1.0.1
6. joblib 0.14.1
7. cython 0.29.15
8. tensorflow 2.2.0
 
 Note: if there are some issues with running cython code, in self_learning.py comment lines 3-4 and change the argument cython to False on line 276.
 
 
 ### Run
 
The experiments for Section 4.1 can be reproduced by running:
>>> bash run_ablation_fsga.sh protein
>>> python export_ablation_fsga.sh protein

The experiments for Section 4.2 can be reproduced by running:
>>> bash run_criteria.sh protein
>>> python export_criteria.sh protein

The experiments for Section 4.3 can be reproduced by running:
>>> bash run_sota.sh protein
>>> python export_sota.sh protein

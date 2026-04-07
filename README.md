# Entropy-Profiled-Diagnostic-Boundary-for-Disease-Pairs-
Version: 0.1.0

Entropy-Profiled Diagnostic Boundary for Disease Pairs Test Used To Quantify Uncertainty Across Disease Pairs. 

*Next Version Will Include A Fully Executable Python Package. 

Instructions to run code:
In the code folder, there are 3 files. 
00_Test_Code.py is the framework for the test. Upon running the code in Python, follow the input scheme. 
01_SyntheticDataSets.py is the code used to generate all 20 datasets.
02_Full_Folder_Run.py is code that repeats file 00_Test_Code.py, over a folder of datasets structured with the same number of antibodies and same definition of classes 0 and 1

Python Packages Used on Local System:

matplotlib                3.10.1
numpy                     2.2.5
pandas                    2.3.3
scikit-learn              1.7.2
scipy                     1.17.0



Abstract:
Bioinformatics increasingly requires analytical approaches capable of capturing uncertainty and complex relationships within high-dimensional biomedical data. To address this limitation, we introduce the Entropy-Profiled Diagnostic Boundary for Disease Pairs, a method that evaluates cohort separation through entropy-based uncertainty profiles. The approach embeds two cohorts within a virtual feature space and evaluates classification certainty along a centroid trajectory using Shannon’s entropy index. The method was evaluated across twenty synthetically generated datasets and one clinical dataset. Across synthetic datasets, the framework produced clear entropy transitions and successfully identified boundary-injected datapoints. Applied to a clinical cohort, the method independently recovered a known diagnostic boundary and identified patients with serological profiles consistent with disease overlap. This work contributes a novel entropy-based framework for analyzing disease cohort boundaries that emphasizes uncertainty profiles and patient-level contributions to classification ambiguity. Rather than functioning as a confirmatory statistical test of association, the method serves as an exploratory analytical tool that characterizes gradients of disease separation while highlighting data points that are often treated as outliers in conventional statistical analyses.
Key Words:
Shannon entropy, Uncertainty quantification, Disease boundary classification, Cohort separation, Autoimmune disease, Diagnostic informatics.


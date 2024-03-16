


Description:

-The file "A2_.py" is the code corresponding to the paraphrase detection instructions for the second assignment.

-The script takes 3 inputs as specified in the assignment guidelines : the train.data , dev.data and test.label files from the dataset available on github: https://github.com/cocoxu/SemEval-PIT2015

- The output will display the classification report as well as 3 visualizations (confusion matrix, ROC curve, Precision-Recall curve) for 4 different models on the dev dataset: Baseline (Algo A), SBERT (Algo B), Fine-Tuned SBERT (Improved Algo B), and Multinomial Naive Bayes (Algo C) . 
It will also display these same visualizations on the Fine-Tuned SBERT model with the test dataset.

- In total, there should be 5 classification reports and 15 visualizations.

In this classification report. Here is an example of a Classification Report:



Multinomial Naive Bayes (Algo C) Classification Report
              precision    recall  f1-score   support

         0.0       0.70      0.96      0.81      3257
         1.0       0.45      0.07      0.12      1470

    accuracy                           0.68      4727
   macro avg       0.57      0.52      0.46      4727
weighted avg       0.62      0.68      0.59      4727




To execute the script:

IMPORTANT NOTE: The script will take approximately 1 hour and 20 minutes to run because we are training the SBERT Model (Improved Algo B)

- Execute the script by entering the command "python A2_.py" in the command prompt.
- Before running the script, you may have to install certain libraries if you don't have them installed already:

pip install sentence_transformers
pip install torch
pip install pandas
pip install scikit-learn
pip install seaborn
pip install transformers
pip install tensorflow-hub
pip install numpy
pip install scipy
pip install matplotlib





main.py - is a seq2seq evaluator with a default question -
'Show the times used by climbers to climb mountains in Country Uganda.'
which the translater predicted correctly.
Kindly input the next question on prompt after running the file if you want to 
test it. 

Seq2Seq_model.py/Seq2Seq.ipynb - is transformer with attntion model built using the Spyder dataset.
I have commented out the training part. However translation of validation and test set
would take some time here. I have used seaborn violin plot to plot cosine similarity vs effectiveness.

Checkpoints - This folder is used to store ceckpoints while training the model and loaded while evaluating

ValidationSetDataFrame.CSV - This Contains Questions from the Validation Set,
The original Query, The Predicted Query, Correctly Predicted Query (if not correctly predicted the cell is empty), Number of Tables Predicted Correctly (0,1,2), 
Cosine Similarity Score and Number of Columns Predicted Correctly/ Total Number of Columns in Original Query ( This is 1 when all columns are correctly predicted)

TestSetDataFrame.CSV - This Contains Questions from the Test Set,
The original Query, The Predicted Query, Correctly Predicted Query (if not correctly predicted the cell is empty), Number of Tables Predicted Correctly (0,1,2), 
Cosine Similarity Score and Number of Columns Predicted Correctly/ Total Number of Columns in Original Query ( This is 1 when all columns are correctly predicted),
Effectiveness based on Cosine Similarity Score
Very Poor - Similarity < 0.2
Poor - Similarity > 0.2 but <= 0.4
Good - Similarity > 0.4 but <= 0.6
Very Good - Similarity > 0.6 but <= 0.8
Excellent - Similarity > 0.8 but <= 1

SimilarityVsEffectiveness.png - Similarity here refers to cosine similarity from 0-1
between the original and the predicted output of the Seq2Seq Model. Effectiveness is a categorization based on the similarity score. 

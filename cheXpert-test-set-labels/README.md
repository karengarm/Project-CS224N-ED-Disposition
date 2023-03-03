# CheXpert test set labels.

When collecting the CheXpert dataset, we collected a test set consisting of 500 studies from 500 patients randomly sampled from the 1000 studies in the report test set. Eight board-certified radiologists individually annotated each of the studies in the test set following the same procedure and post-processing as described for the validation set. The majority vote of 5 radiologist annotations serves as a strong ground truth: 3 of these radiologists were the same as those who annotated the validation set and the other 2 were randomly sampled. The remaining 3 radiologist annotations were used to benchmark radiologist performance.

Here, we are releasing the following:
1. Test set annotations from each of the individual radiologists, divided into the benchmark radiologists, and the groundtruth. The majority vote ground truth is given in groundtruth.csv. The actual images corresponding to the test set are now available here: https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
2. The chexpert and chexbert labeler outputs for these studies.
3. Predictions and decisions made by some of the leaderboard models on these studies. These models were used to support the findings in CheXternal: https://arxiv.org/pdf/2102.08660.pdf

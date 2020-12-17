# DL-Model-for-Sentiment-Analysis-from-audio-files---RAVDESS-DATASET

RAVDESS dataset is a database of emotional speech and song that contains 7365 files.</b>
All other information could be found on the following link:</b>
https://zenodo.org/record/1188976</b>

# Model Design
The model takes an input of MFCC with a shape of (300,200) And feeds this input to a base model</b>
Resnet50 followed by a softmax layer with 6 nodes. The model is trained from scratch.</b>

# Model Performance
The model gave a categorical accuracy of 80% and an AUC of 94.</b>

![AUC](https://github.com/AhmedFakhry47/DL-Model-for-Sentiment-Analysis-from-audio-files---RAVDESS-DATASET/blob/main/Logs%20%2B%20Pretrained%20Model/MFCC_on_RAVDS.png)

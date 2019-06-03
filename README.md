# SK Telecom Deep Learning Tutorial

Google Drive Link  
https://drive.google.com/drive/folders/168a7_v5eTcpfVMAFM7VEwSA7dOnYYGCU?usp=sharing  


Text Classification - https://colab.research.google.com/github/ku-milab/DeepLearningTutorial/blob/master/TF1/text_classification_TF1.ipynb 
````
# Try this! 
for cnt, (dat, lbl) in enumerate(zip(test_data[:10], test_labels[:10])):
    out_tensor = model(dat[None])
    prediction = tf.keras.backend.eval(out_tensor)
    print(decode_review(dat))
    print("Predict %f, Label %d"%(prediction, lbl))
````

Text Generation - https://colab.research.google.com/github/ku-milab/DeepLearningTutorial/blob/master/TF1/text_generation_TF1.ipynb 
Word2Vec - https://colab.research.google.com/github/ku-milab/DeepLearningTutorial/blob/master/TF1/word2vec_TF1.ipynb 
Text Generation - https://colab.research.google.com/github/ku-milab/DeepLearningTutorial/blob/master/TF1/image_captioning_with_attention_TF1.ipynb 
Text Generation - https://colab.research.google.com/github/ku-milab/DeepLearningTutorial/blob/master/TF1/nmt_with_attention_TF1.ipynb  

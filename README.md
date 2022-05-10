# Machine-Learning-RVDM

# If There's A Whale There's a Way

In this project we aim to contribute to the protection and conversation of marine mammals by automating whale and dolphin photo-ID to significantly reduce image identification times.
This project is based on a [Kaggle competition](https://www.kaggle.com/c/happy-whale-and-dolphin/overview) started by "Happywhale", which is a research collaboration and citizen science web platform with the mission to increase global understanding and caring for marine environments through high quality conservation science and education. They provide a dataset that contains images of over 15,000 unique individual marine mammals from 26 different species collected from 28 different research organizations. Images focus on dorsal fins and lateral body views. The challenge is to develop a model that is able to distinguish between unique - but often subtle - characteristics of the natural markings of whales and dolphins. 



## Authors

- Reed Garvin [@skier921](https://www.github.com/skier921)
- Dinah Rabe [@dinahrabe](https://www.github.com/dinahrabe)
- Maren Rieker [@marenrieker](https://www.github.com/marenrieker)
- Victor Möslein [@sailandcode](https://www.github.com/sailandcode)

## How to Access the Data

There are no images saved on the GitHub as folder size would be too big. You need to download the images from [Kaggle](https://www.kaggle.com/datasets/phalanx/whale2-cropped-dataset). Please be aware that you may need to change the path of the images in the code to match your local setup.  

## Some Additional Comments

To run the ML_Classification Notebook, you have to run the ML_Preprocessing Notebook first, as this transforms the images into numerical dataframes and saves them in the respective "input/clean" folder.

To access the segmentation go to preprocessing, than Tracer-Main, there you find the segmentation jupyter notebook. 

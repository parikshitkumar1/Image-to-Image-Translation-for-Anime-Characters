---

# C2
## Motivation
To implement a project based on <a href = "https://arxiv.org/pdf/1611.07004.pdf"> Image-to-Image Translation with Conditional Adversarial Networks </a>, by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros.
## Requirements
Python 3.8 or above with all [requirements](requirements.txt) dependencies installed. To install run:
```python
$ pip3 install -r requirements.txt
```
#### make a folder called "WEIGHTS" in the utils folder and put the weights (https://drive.google.com/file/d/1hLM_ZHTzi7GsQtSL-1bvG5RT2XMMBamI/view?usp=sharing)
  
## To run the app
```python
$ streamlit run app.py
```
#### in case of albumentations error:
```python
$ pip install -U git+https://github.com/albu/albumentations --no-cache-dir
```

### Screenshots
<img src="https://user-images.githubusercontent.com/52780573/123106829-9bbc0a80-d456-11eb-90d4-09d44a0e2ec1.png" data-canonical-src="" width="900" height="500" />
<img src="https://user-images.githubusercontent.com/52780573/123106952-b8584280-d456-11eb-973d-82c234a64501.png" data-canonical-src="" width="900" height="500" />

### Model Components and other details:

Generator: Unmodified UNET

<img src="https://user-images.githubusercontent.com/52780573/123109568-f8b8c000-d458-11eb-9a83-dbf6d5b78cc9.png" data-canonical-src="" width="700" height="400" />


Discriminator: Unmodified PatchGAN

<img src="https://user-images.githubusercontent.com/52780573/123108728-48e35280-d458-11eb-93c1-5df850b3e5e6.png" data-canonical-src="" width="700" height="400" />

#### Dataset:

modified version of https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair by Taebum Kim

#### Hyperparameters:

LEARNING_RATE = 2e-4

BATCH_SIZE = 16

NUM_WORKERS = 2

IMAGE_SIZE = 256

CHANNELS_IMG = 3

L1_LAMBDA = 100

LAMBDA_GP = 10

NUM_EPOCHS = 7


#### final scores: 

D_fake=0.179, D_real=0.859

### Results
#### (256 x 256 x 3 output)



<div>
    <img src="https://user-images.githubusercontent.com/52780573/123104728-d3c24e00-d454-11eb-8a57-f102a9a682eb.jpg" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123104738-d6bd3e80-d454-11eb-874b-ba9fb3f7bf70.png" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123105066-1be17080-d455-11eb-8ff7-17ed40e0f47e.jpg" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123105069-1d129d80-d455-11eb-8392-4f1ca855c68a.png" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123105667-99a57c00-d455-11eb-996a-cd273450ecb1.jpg" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123105672-9ad6a900-d455-11eb-9386-3dfb9d33ec0a.png" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123105799-b6da4a80-d455-11eb-9088-9e31873badb2.jpg" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123105804-b80b7780-d455-11eb-80bf-1e5bc28c41d9.png" width="200" height="200"/>
   
</div>






## Might Do
- [ ] Implement with different datasets
- [ ] 



---

---

# C2
## Motivation
To implement a project based on <a href = "https://arxiv.org/pdf/1611.07004.pdf"> Image-to-Image Translation with Conditional Adversarial Networks </a>, by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros.
## Requirements
Python 3.8 or above with all [requirements](requirements.txt) dependencies installed. To install run:
```python
$ pip3 install -r requirements.txt
```
### make a folder called "WEIGHTS" in the utils folder and put the weights (https://drive.google.com/file/d/1hLM_ZHTzi7GsQtSL-1bvG5RT2XMMBamI/view?usp=sharing)
  
## To run the app
```python
$ streamlit run app.py
```



## -->

#### super-res
<img src="https://user-images.githubusercontent.com/52780573/110626302-90561380-81c6-11eb-9313-8315c1c1d21c.png" data-canonical-src="" width="900" height="500" />

#### non super-res

<img src="https://user-images.githubusercontent.com/52780573/110474334-8ddfb500-8105-11eb-96d7-47cb97f820c9.png" data-canonical-src="" width="900" height="500" />

## Architecture and other details

#### Trained for nearly 150 epochs on approximately 8000 Albrecht Dürer paintings

#### LapSRN_x8 used to upscale paintings by a factor of 8 (pretrained)

<img src="https://user-images.githubusercontent.com/52780573/110354770-8a452300-805e-11eb-817c-3045e33b536a.gif" data-canonical-src="" width="900" height="500" />


#### w1 ---> weights saved at 100 epochs, w2 ---> weights saved at 150 epochs, total epochs ~150


#### Partial Dataset ---> https://www.kaggle.com/ikarus777/best-artworks-of-all-time

#### all images resized to 64 x 64 x 3(channel)

## Results


<div>
    <img src="https://user-images.githubusercontent.com/52780573/123104728-d3c24e00-d454-11eb-8a57-f102a9a682eb.jpg" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/123104738-d6bd3e80-d454-11eb-874b-ba9fb3f7bf70.png" width="200" height="200"/>
    <img src="https://user-images.githubusercontent.com/52780573/110626762-238f4900-81c7-11eb-9b6b-73cd42601306.png" width="100" height="100"/>
    <img src="https://user-images.githubusercontent.com/52780573/110626764-2427df80-81c7-11eb-9dc4-fbc63dd8740d.png" width="100" height="100"/>
    <img src="https://user-images.githubusercontent.com/52780573/110626768-24c07600-81c7-11eb-9f5f-6bbd7bc7bdac.jpeg" width="100" height="100"/>
    <img src="https://user-images.githubusercontent.com/52780573/110626769-24c07600-81c7-11eb-959a-bcf5d5eb70cc.png" width="100" height="100"/>
    <img src="https://user-images.githubusercontent.com/52780573/110627447-ff803780-81c7-11eb-97ab-c0f642509f4d.png" width="100" height="100"/>
    <img src="https://user-images.githubusercontent.com/52780573/110627568-28083180-81c8-11eb-9f77-5797f07fd4c8.png" width="100" height="100"/>
   
</div>

#### finals scores: loss_g: 0.5128, loss_d: 1.1873, real_score: 0.5859, fake_score: 0.0469

<img src="https://user-images.githubusercontent.com/52780573/110355252-07709800-805f-11eb-8816-7e07103fad94.png" data-canonical-src="" width="500" height="350" />


<img src="https://user-images.githubusercontent.com/52780573/110355448-3f77db00-805f-11eb-80d1-d853d1e4140a.png" data-canonical-src="" width="500" height="350" />



## Might Do
- [ ] Upload WGAN-GP Notebook




---

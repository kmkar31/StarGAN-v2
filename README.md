# StarGAN-v2

### Objective: 
  To train a **Single** Generator G which generates diverse images of each domain y corresponding to a given image x.

### Technical Terms used in the paper:
 
  1. **Diverse Images**: Produces different images every time given a source image and a domain. (Disadvantage of StarGAN)
  2. **Domain**:  A set of images forming a visually distinctive category. **Ex:** Gender of a Person
  3. **Style**: Unique Appearance of each image. **Ex:** Hairstyle, Makeup etc.
  4. **Latent Space**: A space over which the compressed data is represented in order to remove extraneous information and find fundamental similarities between two datapoints.      [Understanding Latent Space](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d).

<details><summary><b>Read More</b></summary>

----

**The Network attemps to generate domain-specific style vectors in the learned style space of each domain and train G to reflect these vectors in the output image**

### Components of the Network

  1. **Generator G**: Takes an Input image **x** and a style code **s** to generate an output image. The style code removes the need of providing the domain of the image to G allowing it to generate images of all domains. **s** is designed to represent the style of a specific domain **y**, which removes the necessity of providing y to G and allowing tit to synthesize images of all domains.
  2. **Mapping Network F**: Takes the Domain (**y**) and Latent Code **z** (Gaussian Noise) to generate the style code **s** (which are domain specific).  Diverse style codes can be generated by randomly sampling the latent vector **z** and the domain **y** rndomly
  3. **Style Encoder E**: Takes in an Image **x** and a domain **y** to generate the style code **s** of x.The Style Encoder can produce diffrent style codes using different reference imaages.
  4. **Discriminator D**: Consists of multiple output branches with each branch D<sub>y</sub> classifying whether or not the image is a **real** image belonging to Domain **y**. 
  
  
  ![Network Architecture](https://pythonawesome.com/content/images/2020/01/overview--2-.png)
  
  ----
  
  ### Training Objectives
  #### Notation:
  
   * Original Image - **x**
   * Original Domain - y
   * Target Domain - &#7929;
   * Style Code of the Target Domain predicted by the Mapping Network - &#353;
   * Style Code of the Original Image predicted by the Style Encoder - &#349;
   * Loss - &#120027;
  
  1. **Adversarial Objective  &#120027;<sub>adv</sub>**: 
     * Sample a latent code **z** and a domain **&#7929;** randomly. Generate a style code  **&#353; = F<sub>&#7929;</sub>(z)**
     * Generate an Output image **G(x,s)** using the generated style code.
     * Learn using **Adverserial Loss**. While training the Generator, there is no control over the *log[D<sub>y</sub>(x)]*. So the Generator tries to *Minimise* the expected value of the *log(1-D<sub>&#7929;</sub>(G(x,&#353;)))* term. We want the discriminator to classify the generated image as real with as high a probability as possible. Since log is a monotonically increasing function, minimising the loss would try and maximise this probability. When training the Discriminator, however, we want to *Maximise* the loss to maximise D<sub>y</sub>(x) since **x** truly belongs to the domain **y**
  2. **Style Reconstruction &#120027;<sub>sty</sub>**:
      * To *Minimise* the style Reconstruction loss i.e., to train the **Style Encoder** to correctly predict the style of the image and to push the **Generator** towards greater use of the provided style code. The output of the Style Encoder should ideally be &#349;
  3. **Style Diversification &#120027;<sub>ds</sub>**: 
      * We try to *Maximise* the difference between images generated using two different style codes **&#353;<sub>1</sub>** and **&#353;<sub>2</sub>** produces using two different latent codes **z<sub>1</sub>** and **z<sub>2</sub>**
  4. **Source Characteristics &#120027;<sub>cyc</sub>**:
      * We try to *Minimise* the difference between the original image and the generated output given an image which is generated using **x** and **&#349;** and the style code predicted by the Style Encoder i.e., ensure that the generator preserves characteristics of the original image.
  
 #### Full Objective:
 
 #### min<sub>G,F,E</sub> max<sub>D</sub> &#120027;<sub>adv</sub> + &#955;<sub>sty</sub> &#120027;<sub>sty</sub> - &#955;<sub>ds</sub> &#120027;<sub>ds</sub> + &#955;<sub>cyc</sub> &#120027;<sub>cyc</sub>
  * The &#955;'s are hyperparameters
  
  ### Evaluating the Model:
  
  * **Frechet Inception Distance** - The Fréchet inception distance (FID) is a metric used to assess the quality of images created by the generator of a generative adversarial network (GAN).The FID compares the distribution of generated images with the distribution of real images that were used to train the generator. (Lower FID is better) [FID](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
  * **Learned Perceptual Image Patch Similarity** - A Mesure Of Diversity in generated Images (Higher is Better)
  
### Network Architecture:

#### Layers:
  * **LAYERS**
    * [Convolution and Pooling Layers](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
  * **NORMALIZATIONS**
    * [Instance Normalization](https://becominghuman.ai/all-about-normalization-6ea79e70894b). In Instance Normalization, mean and variance are calculated for each individual channel for each individual sample across both spatial dimensions.
    * [Adaptive Instance Normalization](https://paperswithcode.com/method/adaptive-instance-normalization#:~:text=Adaptive%20Instance%20Normalization%20is%20a,Instance%20Normaliation%20is%20an%20extension). "Aligns" the instance-normalized-sample with the given style code.
  * **ACTIVATIONS**
    * [ReLU and LeakyReLU](https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e)
    

### AdaptiveWing Loss for Robust Face Alignment via Heatmap Regression

#### [Heatmap Regression](https://www.arxiv-vanity.com/papers/1609.01743/#:~:text=The%20proposed%20part%20heatmap%20regression%20is%20a%20CNN%20cascade%20illustrated%20in%20Fig.&text=The%20second%20subnetwork%20is%20a,location%20of%20the%20body%20parts.) 

  

</details>
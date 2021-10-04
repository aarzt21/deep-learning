# deep-learning
This is a interesting deep learning project I have done together with Valanti Karypidis. 
The task was as follows: 
  
  Given three pictures of food (A, B and C), we trained a special convolutional neural network (CNN)
  which could then classify if picture A was closer/more similar in taste and texture to B or C. 
  So basically, we trained a CNN to compare pictures. The our approach was inspired by the so called
  Face-Net paper: https://arxiv.org/abs/1503.03832. 
  
  Once we wrote our custom loss function we used pre-trained weights for the CNN (ResNet50 and VGG16). 
  The final step was to train multiple CNN's and do a majority voting to average out the noise and get
  better predictive performance - which did the trick for cracking the hard baseline. 
  
The project was part of the ETH course "Introduction to Machine Learning" and we scored a straight
6.00 (A+) on this particular project. 

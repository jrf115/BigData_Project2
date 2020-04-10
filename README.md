# BigData_Project2
 Use Keras to create and compare deep learning models for the MNIST Fashion data set. Create saved models to be reused, and create logs for visualizing in TensorBoard. Use the MNIST Fashion data set for training and validation.


# How to run project
Download anaconda for python and create a new tensorflow environment:

	conda create -n tf2 python=3.6
	conda activate tf2
	conda install -c conda-gorge keras
	conda install matplotlib

After installing everything needed run the code and view the code produced logs on TensorBoard:

	conda activate tf2
	python p2_5276.py
	tensorboard --logdir logs/fit
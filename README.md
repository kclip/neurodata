# neuro-data
Code for preprocessing and loading data from neuromorphic datasets.
Part of this code has been used for the following works:

N. Skatchkovsky, H. Jang, and O. Simeone, Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence, accepted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2020.
https://arxiv.org/abs/1910.09594

H. Jang, N. Skatchkovsky, and O. Simeone, VOWEL: A Local Online Learning Rule for Recurrent Networks of Probabilistic Spiking Winner-Take-All Circuits, to be presented at ICPR 2020
https://arxiv.org/abs/2004.09416

N. Skatchkovsky, H. Jang, and O. Simeone, End-to-End Learning of Neuromorphic Wireless Systems for Low-Power Edge Artificial Intelligence, accepted to Asilomar 2020
https://arxiv.org/abs/2009.01527

# Installing 
This code can now be installed as a package and is meant to be eventually shared in pip.
To clone and install locally the package, run 
~~~
git clone https://github.com/kclip/neurodata 
cd neurodata/ 
python -m pip install -e . 
~~~

# Data preprocessing
Scripts to preprocess the MNIST-DVS and DVSGestures dataset are  given in the `preprocessing` module. 
Make sure to first download and then preprocess the dataset using the script in `preprocessing`.

To add your own datasets, save them as an .hdf5 file respecting the current structure: <br />
`
/ root (Group) ` <br />
` /stats (Group) ` <br />
` /stats/test_data (Array) [n_examples_test, n_pixels_per_dim] ` <br />
` /stats/test_label (Array) [n_examples_test, n_classes] ` <br />
` /stats/train_data (Array) [n_examples_train, n_pixels_per_dim] ` <br />
`/stats/train_label (Array) [n_examples_train, n_classes] ` <br />
` /train (Group) ` <br />
` /train/labels (Array) [n_examples_train, 1] ` # indicates labels for each train example <br /> 
` /train/1  (Array) [example_length, 4] ` # indicates event time, x axis position, y axis position, polarity <br />
`...` <br />
` /train/n_examples_train  (Array) [example_length, 4] ` # one array per train example <br />
` /test (Group) ` <br />
` /test/labels (Array) [n_examples_test, 1] ` # indicates labels for each test example <br /> 
` /test/1 (Array) [example_length, 4] ` # indicates event time, x axis position, y axis position, polarity <br />
`...` <br />
` /test/n_examples_test (Array) [example_length, 4] ` # one array per test example <br />




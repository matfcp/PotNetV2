# PotNet v2

This is the code for the *Heritage's special issue* paper [A Deep Learning Approach to Assist in Pottery Reconstruction from Its Sherds](https://www.mdpi.com/2571-9408/8/5/167), by Matheus Pinho, Guilherme Mota and Gilson Costa, presented at the Conference on Cultural Heritage and New Techologies (CHNT) 29th edition.

Here we present a deep-learning-based method to find the relative positions of pottery sherds to help in pottery reconstruction. The neural network architecture used as the backbone of the method, the so-called PotNet, was designed to perform non-linear regressions.

The method starts with 3 virtual vessels broken down thousands of times. The point clouds of the resulting sherds in a standard position are the input to the networks. The targets are the matrices that move the point clouds from this standard position to the expected one (or, at least, as close as possible to it).

Two network branches are trained for each vessel, one for the rotation parameters and the other for the translation values.

## Installation

First start by cloning this repository 
```bash
git clone https://github.com/matfcp/PotNetV2.git
```
Create a [`conda` environment](https://docs.conda.io/projects/conda/en/latest/index.html):
```bash
conda create -n potnet
conda activate potnet
```

Install the dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Training

If you want to process the data, first download the [ZIP](https://mega.nz/file/X8wkjSJQ#iDKT71NUTDCtFyWbU4gYqDTeWVnOr-d03qnSKmlUddQ) file contaning 2000 synthetic breaks of the LV vessel as an example. Just unzip it and put the *LV_dataset* folder inside *data* folder. You can use the *process_data.py* script to generate the processed file that will be used to train the networks. 

You can also download the ready-to-use LV processed data ([points](https://mega.nz/file/H5ZigJrC#14y9DjPuvR5lGsMAnCwqpXL1pn37ALGoZsBzDdLf3z8), [targets](https://mega.nz/file/bp4zAYKQ#AkrTA3XWPs3rwmwfOBTFKaVptAbwwD_nRqmsQrJEzxA)) to run tests, just put the files inside *train_data/LV* folder.

To train, simply open the terminal in the parent directory of this repository and do
```bash
scripts/train.sh
```

In the *train.sh* file you can change the name of the vessel or file you want to use, and also the network branch you want ('trans' or 'rot').

## Testing

The trained models and the training graphs will be saved in *models*. We made available trained models and 5 test synthetic breaks for the LV vessel within *test_example* folder. Simply run 
```bash
scripts/test.sh
```

You can also test the models with the downloaded LV vessel breaks from the previous step, just change the path in the *test.sh* file with the correct path of the test breaks (should be './data/LV_dataset/test/'). You can change the folder of the break in the *test.sh* file.

The files with the predictions will be saved in *results*. We provided some results from the *test_example* data. There's a *files* folder with the .stl predictions (you can save as .ply, just change the name in the *test.py* script), some evaluation metrics for each sherd (MSE: mean squared error of ptc; RMSE: root mean squared error of ptc; RMSE (x,y,z): rmse regarding x,y,z coordinates separately; distCNTRD: height of the centroid along vertical axis; stdPTC: standard deviation of points) with values in meters; and the matrices (unnorm2canon: moves the sherd cloud from it's true position, within the vessel's coordinate system, to canonical position; T_pred: moves the sherd cloud from canon position to predicted position).

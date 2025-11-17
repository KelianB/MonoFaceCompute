ENV_NAME=sapiens_lite

echo "Creating conda environment"
conda create -n $ENV_NAME python=3.10 
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
if echo $CONDA_PREFIX | grep $ENV_NAME
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script."
    exit
fi

echo "Installing dependencies"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python tqdm json-tricks natsort

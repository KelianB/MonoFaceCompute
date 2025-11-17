ENV_NAME=StableNormal

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
pip install transformers==4.36.1 xformers==0.0.21 'diffusers==0.30.3' 'gradio>=4.32.1' 'gradio-imageslider>=0.0.20' pygltflib==1.16.1 trimesh==4.0.5 imageio imageio-ffmpeg Pillow einops==0.7.0 spaces accelerate 'diffusers>=0.28.0' matplotlib==3.8.2 scipy==1.11.4 torch==2.0.1 torchvision
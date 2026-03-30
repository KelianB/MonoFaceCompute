(cd ./submodules/INFERNO/inferno_apps/EMOCA/demos && ./download_assets.sh)
(cd ./submodules/INFERNO/inferno_apps/FaceReconstruction && ./download_assets.sh)
echo "Downloading pre-trained segmentation model"
(mkdir -p ./submodules/face-parsing.PyTorch/res/cp && gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O ./submodules/face-parsing.PyTorch/res/cp/79999_iter.pth)
echo "Downloading pre-trained MODNet model"
(mkdir -p ./submodules/MODNet/pretrained && gdown 1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX -O ./submodules/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt)
echo "Downloading pre-trained SMIRK model"
(mkdir -p ./submodules/SMIRK/pretrained_models && gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O ./submodules/SMIRK/pretrained_models/SMIRK_em1.pt)
# echo "Downloading pre-trained Omnidata normal estimation model"
# (cd submodules/omnidata && sh omnidata_tools/torch/tools/download_surface_normal_models.sh && mv 'pretrained_models/omnidata_dpt_normal_v2.ckpt?download=1' 'pretrained_models/omnidata_dpt_normal_v2.ckpt')
# echo "Downloading pre-trained DSINE normal estimation model"
# (cd submodules/DSINE && mkdir -p ./projects/dsine/checkpoints/exp001_cvpr2024 && gdown 1Wyiei4a-lVM6izjTNoBLIC5-Rcy4jnaC -O ./projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt)
echo "Downloading pre-trained Sapiens normal estimation model"
(cd submodules/sapiens && wget https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2)
echo "Downloading pre-trained IntrinsicAnything albedo model"
(cd submodules/IntrinsicAnything && mkdir -p ./weights/albedo/checkpoints && wget https://huggingface.co/LittleFrog/IntrinsicAnything/resolve/main/albedo/checkpoints/last.ckpt?download=true -O ./weights/albedo/checkpoints/last.ckpt)

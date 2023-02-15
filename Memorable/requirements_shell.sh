
unzip -qq /content/drive/MyDrive/INTERPRETABILITY/Zip\ Files/Detic.zip -d /
unzip -qq /content/drive/MyDrive/INTERPRETABILITY/Zip\ Files/detectron2.zip -d /
unzip -qq /content/drive/MyDrive/INTERPRETABILITY/Zip\ Files/Interpretability.zip -d /content/
unzip -qq /content/drive/MyDrive/INTERPRETABILITY/Zip\ Files/in_painting.zip -d /content

pip install click==7.1.2

pip install -q  -e detectron2

pip install -q  -r Detic/requirements.txt
pip install -q -r Detic/third_party/CenterNet2/requirements.txt

pip install timm
pip install torchsort

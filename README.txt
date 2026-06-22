Docker system update
====================
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list 
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

Image loading
=============
cd /mount/data/alx/TARGZ
docker load -i speech_base_alx.tar.gz
docker create --gpus all -it --name stt-gpu --shm-size=200g  --restart unless-stopped  -v /mount/data:/mount/data jf.apps.local/docker/speech_base:pytorch25.06-py3-k2-alx-bleach  bash
docker start stt-gpu
docker exec -it stt-gpu /bin/bash


UNPACK Musan
============
mkdir -p /mount/data/alx/intermidiate
mv /mount/data/alx/TARGZ/musan.tar.gz /mount/data/alx/intermidiate
cd /mount/data/alx/intermidiate
tar -xzvf musan.tar.gz

Upack Stages 1-5
================
cd /mount/data/alx/TARGZ
mv Stage*.tar.gz ../
cd ..
tar -xzvf Stage1.tar.gz
tar -xzvf Stage2.tar.gz
tar -xzvf Stage3.tar.gz
tar -xzvf Stage4.tar.gz
tar -xzvf Stage5.tar.gz

Upack alignment example
=======================
cd /mount/data/alx/TARGZ
mv alignment_example.tar.gz ../
cd ..
tar -xzvf alignment_example.tar.gz

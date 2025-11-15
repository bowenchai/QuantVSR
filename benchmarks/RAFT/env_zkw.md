conda create -n raft python=3.11
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
conda install matplotlib tensorboard scipy opencv -c pytorch
conda install tqdm
pip install ipdb
cd networks/resample2d_package
python3 setup.py build
python3 setup.py install
cd ../../
bash warp_error.sh

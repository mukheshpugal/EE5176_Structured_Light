git clone https://github.com/autonomousvision/connecting_the_dots.git
cp infer.py connecting_the_dots/
cd connecting_the_dots

# Download dataset
mkdir val_data && mkdir val_data/syn && cd val_data/syn
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hwrzxDKdhlXGptLDae2eMfKjy8jlUZct' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hwrzxDKdhlXGptLDae2eMfKjy8jlUZct" -O val_data.zip && rm -rf /tmp/cookies.txt
unzip val_data.zip
cd ../..

# Download network parameters
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17KadQOpxU6s86epnz7rGONmY9Z-VKPL2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17KadQOpxU6s86epnz7rGONmY9Z-VKPL2" -O net_0099.params && rm -rf /tmp/cookies.txt

# Change to CUDA 10.0 env and install necessary packages
source activate python3
conda update -n base -c defaults conda -y
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch -y
conda install --file requirements.txt -y

# Building pytorch extensions
cd torchext
python setup.py build_ext --inplace
cd ..

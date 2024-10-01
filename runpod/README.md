# Environment setup (with Runpod)

## Basic system tools

On Runpod, you need to install sudo first
```
su -
apt-get update
apt-get install sudo
```

Run
```
chmod +x system_install.sh
./system_install.sh
```

## Git

Configure git
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

## HuggingFace

Configure HuggingFace CLI
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
You will need this for licensed open-source models.

## Conda

Run
```
chmod +x conda_install.sh
./conda_install.sh
```
Then you can create conda environments (fresh or from our `conda_requirements.yml`).
```
conda env create -f conda_requirements.yml
```
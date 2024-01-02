#!usr/bin/env sh

# To run : curl https://raw.githubusercontent.com/cs50victor/splatter/main/setup.sh -sSf | sh


sudo apt update

# Install git & unzip

sudo apt install git unzip

# Install Cargo
curl https://sh.rustup.rs -sSf | sh

# download repo
mkdir dev

cd dev

# Download and Unzip Models

cd splatter

wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip

unzip models.zip -d models

# Run example

cargo run --example showcase -- models/garden/point_cloud/iteration_7000/point_cloud.ply

#!/bin/bash

# Function to check if conda is installed
check_conda_installed() {
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
        return 0
    else
        echo "Conda is not installed."
        return 1
    fi
}

# Function to install Miniconda
install_miniconda() {
    echo "Downloading Miniconda installer..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest.sh
    echo "Running Miniconda installer..."
    bash Miniconda3-latest.sh -b -p $HOME/miniconda
    echo "Initializing Miniconda..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
}

# Function to create a new conda environment and execute a Python script
run_setup() {
    echo "Creating a new conda environment..."
    conda create -n myenv python=3.8 -y
}

# Main script execution
if check_conda_installed; then
    run_setup
else
    read -p "Conda is not installed. Do you want to install it? (yes/no): " response
    if [[ "$response" == "yes" ]]; then
        install_miniconda
        run_setup
    else
        echo "Conda installation aborted. Exiting."
        exit 1
    fi
fi
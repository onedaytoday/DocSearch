#!/bin/bash

# Define paths
BASE_IMAGE_NAME="my-custom-python-cuda"
BASE_IMAGE_PATH="./DockerBase/$BASE_IMAGE_NAME.tar"
DOCKERFILE="./DockerBase/Dockerfile"
REQUIREMENTS_FILE="./requirements.txt"

# Step 1: Check if the image exists and if the Dockerfile or requirements.txt has been modified
echo "Checking if the base image needs to be rebuilt..."

# Check if the tarball exists and if the Dockerfile or requirements.txt has been modified
if [ ! -f "$BASE_IMAGE_PATH" ] || [ "$DOCKERFILE" -nt "$BASE_IMAGE_PATH" ] || [ "$REQUIREMENTS_FILE" -nt "$BASE_IMAGE_PATH" ]; then
    echo "Base image needs to be rebuilt."

    # Build the custom base image
    docker build -t $BASE_IMAGE_NAME -f $DOCKERFILE ./DockerBase

    # Check if the build was successful
    if [ $? -eq 0 ]; then
        echo "Base image built successfully."

        # Save the image as a tarball inside the DockerBase folder
        docker save -o "$BASE_IMAGE_PATH" $BASE_IMAGE_NAME
        echo "Base image saved to $BASE_IMAGE_PATH"
    else
        echo "Base image build failed. Exiting..."
        exit 1
    fi
else
    echo "Base image is up-to-date."
fi

# Step 2: Build the app image using the custom base image
echo "Building the app image (llm2vec_filesearch)..."
docker build -t llm2vec_filesearch .

# Check if the app image build was successful
if [ $? -eq 0 ]; then
    echo "App image 'llm2vec_filesearch' built successfully."
else
    echo "App image build failed. Exiting..."
    exit 1
fi

echo "Both images have been built successfully!"

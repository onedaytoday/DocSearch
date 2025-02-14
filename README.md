# File Search Project

This project utilizes Docker to build and run the application within an isolated container environment.

## Prerequisites

Before proceeding, ensure the following are installed on your system:

- [Docker](https://www.docker.com/get-started)

## Building and Running the Docker Image

To build and run the Docker image, execute the following command:

```bash
docker build -t file_search_project . && docker run --rm file_search_project

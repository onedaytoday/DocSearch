# File Search Project

This project uses Docker to build and run the application in an isolated container environment.

## Prerequisites

Before running the commands, make sure you have the following installed:
- [Docker](https://www.docker.com/get-started)

## Build and Run the Docker Image

To build and run the Docker image, execute the following command:

```bash
docker build -t file_search_project . | docker run --rm file_search_project

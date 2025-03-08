# Use the custom base image you built
FROM my-custom-python-cuda:latest

# Add any other layers or your app code
WORKDIR /app
COPY . /app

CMD ["python", "main.py"]
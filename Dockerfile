# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application
COPY . .

# Define environment variable
# ENV NAME=Value

# Run your script when the container launches
CMD ["python", "src/train.py"]

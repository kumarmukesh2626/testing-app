FROM python:3.8-slim-buster

# Copy the entire current directory into the /app directory of the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Upgrade Cython before installing other dependencies
RUN pip install --upgrade cython

# Update package lists and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    antiword \
    unrtf \
    libgl1-mesa-glx \
    libglib2.0-0

# Upgrade pip before installing Python dependencies
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5003 for Flask
EXPOSE 5008

# Set the command to run the Flask app
CMD ["python", "app.py"]

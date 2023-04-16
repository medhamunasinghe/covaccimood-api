# Base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements/requirements.txt .

# install pip
RUN pip install pip==23.0.1

# Install dependencies
RUN pip install -r requirements.txt
RUN pip --no-cache-dir install torch

# Copy the project directory to the working directory
COPY . .
COPY BertCNN_bestModel.bin .
RUN ls
RUN pip show torch
RUN echo "ls was here"

# Expose port 8000 for the application
EXPOSE 8000

# Run the application using uvicorn
CMD ["uvicorn", "sentiment_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
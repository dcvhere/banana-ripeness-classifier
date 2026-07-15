# 1. Use the highly stable Python 3.11 image (Solves the TensorFlow mismatch)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your entire repository into the container
COPY . .

# 4. Install system tools and Python dependencies
# We use --no-cache-dir to keep the container size small
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tell the container to listen on the port Cloud Run assigns
ENV PORT=8080
EXPOSE 8080

# 6. The command to boot your app
CMD ["sh", "-c", "streamlit run app/app.py --server.port=${PORT} --server.address=0.0.0.0"]

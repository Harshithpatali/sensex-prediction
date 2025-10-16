# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app and models
COPY . .

# Set environment variables
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# Expose port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8080", "--server.headless=true"]

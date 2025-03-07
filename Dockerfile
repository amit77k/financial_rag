# Use an official lightweight Python image

FROM python:3.10

# Set the working directory in the container

WORKDIR /financial_rag

# Copy the requirements file and install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app to the container

COPY . .

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "financial_rag.py", "--server.port=8501"]

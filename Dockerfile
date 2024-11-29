FROM python:3.12-slim
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080

# Set environment variable to specify the port for Cloud Run
ENV PORT 8080

CMD ["uvicorn", "api:app", "--host=0.0.0.0", "--port=8080"]
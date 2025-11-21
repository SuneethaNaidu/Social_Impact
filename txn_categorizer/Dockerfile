FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements-txn.txt
EXPOSE 8001
CMD ["python", "api.py"]

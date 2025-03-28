FROM python:3.9-slim-buster
WORKDIR /service
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT [ "python3","app.py" ]
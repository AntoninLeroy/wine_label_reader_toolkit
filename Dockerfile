FROM python:3.8

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -qqy \
        tesseract-ocr \
        libtesseract-dev

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["app.py"]
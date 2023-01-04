FROM python:3.7.15


RUN pip install transformers
RUN pip install fastai
RUN pip install flask
RUN pip install gunicorn

WORKDIR /app
COPY . /app

CMD ["gunicorn", "-b", ":8000", "--timeout", "6000", "--workers=10", "app:app"]
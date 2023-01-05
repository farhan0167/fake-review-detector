FROM python:3.7.15


RUN pip install transformers
RUN pip install fastai
RUN pip install flask
RUN pip install gunicorn
RUN pip install -U flask-cors

WORKDIR /app
COPY . /app

CMD ["gunicorn", "-b", ":8000", "--timeout", "6000", "app:app"]
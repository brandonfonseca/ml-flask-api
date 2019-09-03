FROM python:3.6

#update
RUN apt-get update

#install requirements
COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY . /api
WORKDIR /

CMD ["gunicorn", "-w", "1", "-b", ":8000", "-t", "360", "--reload", "api.wsgi:app"]
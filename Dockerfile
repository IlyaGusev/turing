FROM python:3.6
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y git

RUN pip3 install -U pip

RUN mkdir /code
ADD . /code/
WORKDIR /code

RUN pip3 install -r /code/requirements.txt
RUN chmod 777 -R /code

VOLUME  ["/code/data"]
CMD ["/code/run.sh"]

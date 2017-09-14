FROM python:3.6
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y git

RUN pip3 install -U pip

RUN mkdir /module
ADD . /module/
WORKDIR /module

RUN pip3 install -r /module/requirements.txt
RUN chmod 777 -R /module

VOLUME  ["/module/submitions"]
CMD ["/module/run.sh"]

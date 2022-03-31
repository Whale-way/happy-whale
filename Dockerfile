# Start from python slim-buster docker image
FROM python:3.8.6-slim-buster
#Update base packages
RUN apt-get update
RUN apt-get upgrade -y
# Change TimeZone
ENV TZ=Europe/Brussels
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo $(date)
# Copy files to working directory
COPY ./src/ /app/src/
WORKDIR /app
#Install python packages using requirements.txt
RUN pip install -r src/requirements.txt


# Run the script
CMD python src/script.py
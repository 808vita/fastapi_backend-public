#official Python base image.
FROM python:3.13.1
# FROM ubuntu/python:3.12-24.04_stable

#Set the current working directory to /code.
#This is where we'll put the requirements.txt file and the app directory.
WORKDIR /code


#Copy the file with the requirements to the /code directory.
#Copy only the file with the requirements first, not the rest of the code.
#As this file doesn't change often, Docker will detect it and use the cache for this step, enabling the cache for the next step too.
COPY ./requirements.txt /code/requirements.txt


#Install the package dependencies in the requirements file.
#The --no-cache-dir option tells pip to not save the downloaded packages locally, as that is only if pip was going to be run again to install the same packages, but that's not the case when working with containers.
#Note
#The --no-cache-dir is only related to pip, it has nothing to do with Docker or containers.
#The --upgrade option tells pip to upgrade the packages if they are already installed.
#Because the previous step copying the file could be detected by the Docker cache, this step will also use the Docker cache when available.
#Using the cache in this step will save you a lot of time when building the image again and again during development, instead of downloading and installing all the dependencies every time.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN playwright install

RUN playwright install-deps    

#Copy the ./app directory inside the /code directory.
#As this has all the code which is what changes most frequently the Docker cache won't be used for this or any following steps easily.
#So, it's important to put this near the end of the Dockerfile, to optimize the container image build times.
COPY ./app /code/app


#Set the command to use fastapi run, which uses Uvicorn underneath.
#CMD takes a list of strings, each of these strings is what you would type in the command line separated by spaces.
#This command will be run from the current working directory, the same /code directory you set above with WORKDIR /code.
#CMD [ "fastapi","run","app/main.py","--port","80" ]
CMD [ "fastapi","run","app/main.py","--port","80" ]



#$ terminal - build docker image
#docker build -t myimage .

#Notice the . at the end, it's equivalent to ./, it tells Docker the directory to use to build the container image.
#In this case, it's the same current directory (.).

#Start the Docker Container
#$ terminal - Start the Docker Container
#docker run -d --name mycontainer -p 80:80 myimage
# docker run -d --name mycontainer2 --env-file .env   -p 80:80 myimage 

#go to http://192.168.99.100/docs or http://127.0.0.1/docs (or equivalent, using your Docker host).



#$ terminal
#Show both running and stopped containers 
# docker ps -a

#docker container stop
#docker stop mycontainer

# list docker images 
#docker images


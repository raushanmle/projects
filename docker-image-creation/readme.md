<h1 align="center">Docker Commands Cheat Sheet</h1>

## Run a new Container
#### Start a new Container from an Image
Command: `docker run IMAGE`

Example: `docker run python-fastapi`

#### Start a new Container from an Image and assign name

Command: `docker run --name CONTAINER IMAGE `

Example: `docker run --name web python_fastapi`

#### Start a new Container from an Image and map a port 

Command: `docker run --p HOSTPORT:CONTAINERPORT IMAGE `

Example: `docker run --p 8000:8000 python-fastapi`

#### Start a new Container from an Image and map a port 

Command: `docker run --p IMAGE `

Example: `docker run --p python-fastapi`

#### Run a container in background

Command: `docker run -d IMAGE `

Example: `docker run -d python-fastapi`

#### Run a container and assign hostname

Command: `docker run --hostname HOSTNAME IMAGE `

Example: `docker run --hostname raushan python-fastapi`

#### Run a container and add a DNS entry

Command: `docker run --add-host HOSTNAME:IP IMAGE`

Example: `docker run --add-host raushan:192.168.1.100 python-fastapi`

#### Run a container and map local directory into container

Command: `docker run --v HOSTDIR:TARGETDIR IMAGE`

Example: `docker run --v ~/:/usr/share/python-fastapi/fastapi python-fastapi`

## Manage Containers In Docker

#### Show running container

Command: `docker ps`

Show all container running/stopped everything

Command: `docker ps -a`

#### Delete a container

Command: `docker rm CONTAINER`

Example: `docker rm python_fastapi`


#### Delete a container while running

Command: `docker rm -f CONTAINER`

Example: `docker rm -f python_fastapi`

#### Delete stopped container

Command: `docker container prune`

#### Stop a running container

Command: `docker stop CONTAINER`

Example: `docker stop python_fastapi`

#### Start a stopped container

Command: `docker start CONTAINER`

Example: `docker start python_fastapi`

#### Rename a container

Command: `docker rename OLD_NAME NEW_NAME`

Example: `docker rename python_fastapi happy_matsumoto`

#### Copy a file from a container to host

Command: `docker cp CONTAINER:SOURCE TARGET`

Example: `docker cp carsharing.py dreamy_lamarr:fastapi-app/carsharing.py`

#### Copy a file from a host to container

Command: `docker cp CONTAINER:SOURCE TARGET`

Example: `docker cp happy_matsumoto:fastapi-app/carsharing.py carsh.py`

#### Create an image from a container

Command: `docker commit CONTAINER NEW_IMAGE_NAME`

Example: `docker commit happy_matsumoto fastapi_updated`

## Manage Images In Docker

#### Download an Image

Command: `docker pull IMAGE[:TAG]`

Example: `docker pull redis`

Example: `docker pull redis:7.0`


#### Push an Image
First you need to tag to your repository then push
Command: `docker tag IMAGE_NAME USERID/REPONAME:TAGE_NAME`

Example: `docker tag python-fastapi raushan/mlconcept:python-fastapi-1.0`

Command: `docker push USERID/REPONAME:TAGE_NAME`

Example: `docker push raushan/mlconcept:python-fastapi-1.0`


#### show all images

Command: `docker images -a`

#### Delete an image

Command: `docker rmi IMAGE`

Example: `docker rmi python-fastapi`    

#### Delete danglinging images

Command: `docker image prune`

#### Delete all unused images

Command: `docker image prune -a`

#### Build an Image from a Dockerfile

Command: `docker build DIRECTORY`

Example: `docker build .`

#### Build an Image from a Dockerfile

Command: `docker build DOCKER_FILE_DIRECTORY`

Example: `docker build .`


#### Tag an Image

Command: `docker tag image IMAGE_ID REPOSITORY:TAG`

Example: `docker tag f5270bb python-fastapi:version-2`

#### Build an image from file and Tag

Command: `docker build -t IMAGE_NAME DOCKER_FILE_DIRECTORY`

Example: `docker build -t fastapi .`

#### Save an Image to .tar file

Command: `doacker save IMAGE_NAME > FILE_NAME.tar`

Example: `docker save 3ceffe9 > redis.tar`

#### Load an Image from .tar file

Command: `doacker load FILE_NAME.tar`

Example: `docker load -i redis.tar`









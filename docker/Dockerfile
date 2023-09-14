FROM ubuntu:20.04

# Create a non-root user, standard practice for security reasons
ARG username=spk_user
ARG uid=1000
ARG gid=100

ENV HOME /home/$username

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $uid \
    --gid $gid \
    --home $HOME \
    $username

COPY ../environment.yml /tmp/
COPY ../requirements.txt /tmp/
RUN chown $uid:$gid /tmp/environment.yml /tmp/requirements.txt

# Install miniconda
USER $username
ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3

RUN apt-get update && apt-get install -y wget

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Make conda activate command available
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash

# Create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# Build the conda environment
ENV ENV_PREFIX $PWD/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

# Set up a script to activate the environment when the container starts
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $ENV_PREFIX" >> ~/.bashrc

USER $username
CMD [ "/bin/bash" ]

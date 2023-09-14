# Use a more recent Ubuntu version
FROM ubuntu:20.04

SHELL [ "/bin/bash", "--login", "-c" ]

ARG username=spk_user
ARG uid=1000
ARG gid=100
ENV USER $username
ENV HOME /home/$USER
# Fixed the typo here
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $uid \
    --gid $gid \
    --home $HOME \
    $USER

COPY ../environment.yml /tmp/requirements.txt /tmp/
RUN chown $uid:$gid /tmp/environment.yml /tmp/requirements.txt

COPY docker/entrypoint.sh /usr/local/bin/

RUN chown $uid:$gid /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

# install miniconda
USER $USER
ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O $HOME/miniconda.sh && \
    chmod +x $HOME/miniconda.sh && \
    $HOME/miniconda.sh -b -p $CONDA_DIR && \
    rm $HOME/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> $HOME/.profile
RUN conda init bash

ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

ENV ENV_PREFIX $PWD/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

RUN conda run -n env python -m pip install -r /tmp/requirements.txt

ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

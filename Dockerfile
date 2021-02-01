# FROM tensorflow/tensorflow:1.14.0-gpu-py3
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
# FROM araffin/stable-baselines

#https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ARG DEBIAN_FRONTEND=noninteractive

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y software-properties-common && apt-get update -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev\
    git \
    bzip2 \
    python3.6 \
    python3-pip \
    python3-setuptools \
    python3-setuptools-git \
    python3.6-dev \
    xvfb \
    ffmpeg \
    ufw \
    wget \
    freeglut3-dev \
    libgtk2.0-dev \
    libglib2.0-0 \
    libopenmpi-dev \
    zlib1g-dev \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    python3.6-tk && \
    rm -rf /var/lib/apt/lists/*
 
# Install and configure ssh server
# https://docs.docker.com/engine/examples/running_ssh_service/
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server vim nano htop xauth
RUN echo 'PermitRootLogin yes\nSubsystem sftp internal-sftp\nX11Forwarding yes\nX11UseLocalhost no' > /etc/ssh/sshd_config
EXPOSE 22
RUN groupadd sshgroup
RUN useradd -ms /bin/bash -g sshgroup duckie
RUN echo 'duckie:dt2020' | chpasswd
RUN echo 'root:dt2020' | chpasswd

# Build and install nvtop
RUN apt-get update && apt-get install -y cmake libncurses5-dev git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /work/*
# DCMAKE_LIBRARY_PATH fix --> see: https://github.com/Syllo/nvtop/issues/1
# RUN cd /tmp && \
#     git clone https://github.com/Syllo/nvtop.git && \
#     mkdir -p nvtop/build && cd nvtop/build && \
#     cmake .. -DCMAKE_LIBRARY_PATH="/usr/local/cuda-9.0/targets/x86_64-linux/lib" && \
#     make && \ 
#     make install && \
#     cd / && \
#     rm -r /tmp/nvtop
    
# Install and configure screen
RUN apt-get update && apt-get install -y --no-install-recommends screen
COPY .screenrc /root/.screenrc
COPY .screenrc /home/duckie/.screenrc

# Expose ports for tensorboard
EXPOSE 7000
EXPOSE 7001

#RUN rm -r /workspace; mkdir /workspace
#COPY requirements.txt /workspace
#RUN python3.6 -m pip install --upgrade pip setuptools wheel
#RUN pip3.6 install -r /workspace/requirements.txt
# RUN pip3 install -e git://github.com/duckietown/gym-duckietown.git@aido2#egg=gym-duckietown
#RUN git clone https://github.com/duckietown/gym-duckietown.git --branch daffy
#RUN pip install -r gym-duckietown/requirements.txt
#COPY maps/* gym-duckietown/src/gym_duckietown/maps/
#RUN pip3.6 install -e gym-duckietown
#COPY maps/* /usr/local/lib/python3.6/dist-packages/duckietown_world/data/gd1/maps/

WORKDIR /home/duckie
USER duckie
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/duckie/miniconda3
RUN rm Miniconda3-latest-Linux-x86_64.sh
USER root
RUN ln -s /home/duckie/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /home/duckie/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate dtaido5" >> ~/.bashrc && \
    echo "conda activate dtaido5" >> /home/duckie/.bashrc
SHELL ["/home/duckie/miniconda3/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]
COPY "environment_aido5.yml" Duckietown-RL/
RUN chown -R duckie:sshgroup /home/duckie/Duckietown-RL/
USER duckie
RUN conda env create -f Duckietown-RL/environment_aido5.yml
SHELL ["/home/duckie/miniconda3/bin/conda", "run", "-n", "dtaido5", "/bin/bash", "-c"]
RUN git clone https://github.com/duckietown/gym-duckietown.git --branch v6.0.25
RUN pip install -e gym-duckietown
COPY maps/*.yaml gym-duckietown/src/gym_duckietown/maps/
COPY maps/*.yaml /home/duckie/miniconda3/envs/dtaido5/lib/python3.6/site-packages/duckietown_world/data/gd1/maps/           
COPY config Duckietown-RL/config
COPY duckietown_utils Duckietown-RL/duckietown_utils
COPY experiments Duckietown-RL/experiments
COPY maps Duckietown-RL/maps
COPY tests Duckietown-RL/tests
COPY artifacts Duckietown-RL/artifacts
COPY conda_setup_aido5.sh Duckietown-RL/
RUN echo "cd Duckietown-RL" >> /home/duckie/.bashrc

USER root
RUN chown -R duckie:sshgroup /home/duckie/Duckietown-RL/
RUN chown -R duckie:sshgroup /home/duckie/gym-duckietown/src/gym_duckietown/maps/
RUN chown -R duckie:sshgroup /home/duckie/miniconda3/envs/dtaido5/lib/python3.6/site-packages/duckietown_world/data/gd1/maps  
RUN mkdir /var/run/sshd
CMD ["/usr/sbin/sshd", "-D"]
#CMD service ssh start && /bin/bash

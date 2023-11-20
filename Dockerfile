FROM continuumio/miniconda3 AS build

COPY mexpose.yml .

RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda env create -f mexpose.yml

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

RUN conda-pack -n mexpose -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack

# Runtime-stage
FROM ubuntu:22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV XAUTHORITY=/root/.Xauthority

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# Install packages
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    libmysqlclient-dev \
    openjdk-8-jdk \
    pip \
    curl \
    unzip \
    git \
    subversion

# Add Firefox PPA and install Firefox
RUN add-apt-repository ppa:mozillateam/ppa -y && \
    printf 'Package: * \nPin: release o=LP-PPA-mozillateam \nPin-Priority: 1001 \n' | tee /etc/apt/preferences.d/mozilla-firefox && \
    apt-get install -y firefox && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Download and install Fiji
RUN curl -L https://downloads.micron.ox.ac.uk/fiji_update/mirrors/fiji-latest/fiji-linux64.zip -o /tmp/fiji-linux64.zip && \
    unzip /tmp/fiji-linux64.zip -d /home && \
    rm /tmp/fiji-linux64.zip

# Pull MeXpose github repository and IMC-Plugins
RUN cd /home && \
    git clone https://github.com/Luke-Br/MeXpose.git && \
    svn export https://github.com/BodenmillerGroup/ImcPluginsCP/trunk/plugins

# Update .bashrc
RUN echo "source /home/MeXpose/aliases.sh" >> /root/.bashrc && \
    echo "source /home/MeXpose/fiji_macro_launch.sh" >> /root/.bashrc && \
    echo "source /venv/bin/activate" >> /root/.bashrc

CMD ["/bin/bash", "-i"]

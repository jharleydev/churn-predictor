FROM gitpod/workspace-full:latest

RUN curl -fsSL https://get.pulumi.com | sh -s  && \
    sudo -H /usr/bin/pip3 install pulumi_kubernetes flask pulumi && \
    pip3 install pulumi_kubernetes flask pulumi && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    npm install -g typescript

RUN brew update && brew install azure-cli

# Install .NET SDK (Current channel)
# Source: https://docs.microsoft.com/dotnet/core/install/linux-scripted-manual#scripted-install
RUN mkdir -p /home/gitpod/dotnet && curl -fsSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel Current --install-dir /home/gitpod/dotnet
ENV DOTNET_ROOT=/home/gitpod/dotnet
ENV PATH=$PATH:/home/gitpod/dotnet

ENV PATH="${PATH}:/home/gitpod/.pulumi/bin"
ENV PIP_USER=no

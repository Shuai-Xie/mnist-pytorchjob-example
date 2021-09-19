FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /opt/mnist
COPY . .

# docker build -f Dockerfile . -t shuaix/pytorch-dist-mnist:1.0
# docker push shuaix/pytorch-dist-mnist:1.0
# docker pull shuaix/pytorch-dist-mnist:1.0

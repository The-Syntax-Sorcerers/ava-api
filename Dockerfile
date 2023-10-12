# syntax=docker/dockerfile:1

FROM public.ecr.aws/lambda/python:3.9
ENV DOCKER_BUILDKIT=1
RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy function code
COPY . ./
COPY main.py ${LAMBDA_TASK_ROOT}
COPY .env ${LAMBDA_TASK_ROOT}
# COPY models ${LAMBDA_TASK_ROOT}/models

# Install the specified packages
RUN pip install -r requirements.txt

# Copy the environment variables file
# RUN --mount=type=secret,id=mysecret cat /run/secrets/mysecret

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.handler" ]

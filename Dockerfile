# syntax=docker/dockerfile:1

FROM public.ecr.aws/lambda/python:3.9
ENV DOCKER_BUILDKIT=1

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy function code
COPY . ./
COPY w2v_model model_weights ${LAMBDA_TASK_ROOT}


# Install the specified packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir  -r requirements.txt

# Copy the environment variables file
# RUN --mount=type=secret,id=mysecret cat /run/secrets/mysecret


# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
WORKDIR /
CMD [ "main.handler" ]

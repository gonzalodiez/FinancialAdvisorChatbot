FROM python:3.10.12 as base

COPY ./ /src

WORKDIR /src

RUN pip install --upgrade pip 
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pysqlite3-binary

FROM base as build

EXPOSE 8000

ENTRYPOINT ["chainlit", "run", "/src/main.py"]

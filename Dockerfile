FROM python:3.8.0
COPY . /app
WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz
RUN pip install -r requirements.txt
WORKDIR /app/Python
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["CreateDecisionTree.py","--server.maxUploadSize=5"]
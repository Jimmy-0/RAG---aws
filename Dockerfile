FROM python:3.11
EXPOSE 8084
WORKDIR /app
COPY req.txt ./
RUN pip install -r req.txt 
COPY . ./
ENTRYPOINT [ "streamlit","run","app_v2.py","--server.port=8084","--server.address=0.0.0.0" ]

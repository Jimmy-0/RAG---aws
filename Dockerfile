FROM python:3.11
EXPOSE 8083
WORKDIR /app
COPY req.txt ./
RUN pip install -r req.txt 
COPY . ./
ENTRYPOINT [ "streamlit","run","admin_auth.py","--server.port=8083","--server.address=0.0.0.0" ]

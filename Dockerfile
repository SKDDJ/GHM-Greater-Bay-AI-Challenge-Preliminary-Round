FROM skddj/xiugo:v1

WORKDIR /workspace

ENV CONDA_DEFAULT_ENV=uni

COPY indocker_shell.sh ./indocker_shell.sh 

COPY sample.sh ./sample.sh 

COPY score.py ./score.py
 

COPY . . 

CMD ["/bin/bash"] 

# ENTRYPOINT ["/bin/bash", "-c", "conda activate uni"]

RUN echo "chmod -R 777 /workspace/indocker_shell.sh" >> /root/.bashrc

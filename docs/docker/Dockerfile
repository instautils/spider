FROM python:2

COPY . .

RUN chmod +x .venv/bin/activate && ./.venv/bin/activate

CMD [".venv/bin/python", "main.py"]
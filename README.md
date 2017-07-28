# Установка Docker и docker-compose
curl -sSL https://get.docker.com/ | sh
curl -L "https://github.com/docker/compose/releases/download/1.10.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
# Запуск
docker-compose up

# Файлы класть в data/test, забирать final_answer.csv из data
# README2: Полное руководство от развёртывания до работы

Этот документ описывает полный путь: от подготовки удалённого сервера и скачивания проекта до обучения моделей, запуска FastAPI-сервиса и веб-интерфейса, тестирования и опционального Docker-деплоя.

## 1. Предварительные требования
- **ОС:** Ubuntu 22.04 LTS или любая Linux x86_64 с Python ≥ 3.10.
- **Установленные пакеты:** `git`, `python3`, `python3-venv`, `python3-pip`, `build-essential`. На Ubuntu:
  ```bash
  sudo apt update && sudo apt install -y git python3 python3-venv python3-pip build-essential
  ```
- **(Опционально) GPU:** Для тренировки трансформеров потребуется CUDA-драйвер и PyTorch с поддержкой GPU.
- **Порты:** 8000 (FastAPI + UI). Для Docker также понадобятся 3000/8080, если измените прокси.

## 2. Клонирование репозитория на сервер
```bash
cd /opt  # или другая директория для кода
sudo git clone https://example.com/ML-Web.git
sudo chown -R $USER:$USER ML-Web
cd ML-Web
```

## 3. Настройка Python-окружения
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3.1 Установка зависимостей проекта
```bash
make install  # эквивалент pip install -r requirements.txt
```
Если доступа в интернет нет, установите зеркала PyPI или перенесите wheel-файлы вручную.

## 4. Подготовка данных и быстрый осмотр
1. Ознакомьтесь с примером `data/sample_reviews.csv`.
2. При необходимости замените файлы в каталоге `data/` своими размеченными отзывами.
3. Запустите разведочный анализ:
   ```bash
   make eda
   ```
   Отчёт появится в `reports/eda_summary.json`.

## 5. Обучение моделей
### 5.1 Бейзлайн TF-IDF + Logistic Regression
```bash
make train
```
Файлы `models/baseline.joblib` и `models/metadata.json` будут созданы автоматически.

### 5.2 Трансформер (RuBERT/ruRoBERTa и т.п.)
```bash
make train-transformer
```
Скрипт `ml/train_transformer.py` скачает выбранную модель Hugging Face (по умолчанию `cointegrated/rubert-tiny`), дообучит её на ваших данных и сохранит веса в `models/transformer/`.

> Backend при старте проверяет наличие трансформера. Если он есть — используется он, иначе загружается `baseline.joblib`. Если ни одного артефакта нет, сервис работает на `KeywordFallbackModel`, чтобы API оставалось доступным.

## 6. Оценка качества
```bash
make evaluate
```
- Результаты (`accuracy`, `macro_f1`, `classification_report`, `confusion_matrix`) сохраняются в `reports/eval_metrics.json`.
- Эндпоинт `/reports/metrics` и панель «Качество модели» на UI читают именно этот файл.

## 7. Активное обучение и обратная связь
- Пользовательские исправления отправляются на `POST /feedback` и пишутся в `data/feedback.jsonl`.
- Чтобы превратить их в новый датасет:
  ```bash
  make feedback-export  # создаст CSV в data/feedback_dataset.csv
  ```
- После доразметки объедините файл с основной выборкой и повторите обучение.

## 8. История предсказаний и отчёты
- Все запросы `POST /predict` и `POST /predict_file` логируются в `data/prediction_history.jsonl`.
- Для сводной статистики запустите:
  ```bash
  make history-report
  ```
  Отчёт сохраняется в `reports/history_summary.json` и доступен по `/reports/history` + на панели «Сводка истории обращений».

## 9. Запуск тестов
```bash
make test
```
Это unit-тесты для вспомогательных компонентов (хранилище фидбэка, статистика, модель-фоллбэк, загрузчик отчётов).

## 10. Настройка окружения (.env)
Создайте файл `.env` (можно скопировать `.env.example`, если добавите его) и пропишите пути/параметры:
```
APP_MODEL_PATH=models/baseline.joblib
APP_TRANSFORMER_DIR=models/transformer
APP_FEEDBACK_PATH=data/feedback.jsonl
APP_HISTORY_PATH=data/prediction_history.jsonl
APP_EVAL_METRICS_PATH=reports/eval_metrics.json
APP_HISTORY_SUMMARY_PATH=reports/history_summary.json
APP_ALLOW_ORIGINS=["*"]
```
Все значения имеют дефолты, поэтому .env требуется только при кастомизации.

## 11. Запуск FastAPI + фронтенда
### 11.1 Локально
```bash
make serve
# или
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Проверьте эндпоинты:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
     -d '{"text": "Ваш сервис работает отлично!"}'
```
Интерфейс доступен на `http://SERVER_IP:8000/ui`.

### 11.2 Работа с CSV прямо из UI
1. Откройте «Пакетная классификация CSV».
2. Загрузите файл формата `text,...` (минимум колонка `text`).
3. Получите сводку по классам и скачайте результаты.

## 12. Docker-деплой
```bash
docker compose up --build -d  # собирает backend + uvicorn и отдаёт UI
```
Файлы моделей должны находиться внутри контейнера в `/app/models`. Скопируйте их заранее:
```bash
docker cp models/baseline.joblib ml-web-backend-1:/app/models/
```
Переменные среды можно переопределить в `docker-compose.yml` или `.env`.

## 13. Обновление модели без простоя
1. Обучите новую модель (baseline или трансформер).
2. Прогоните `make evaluate` и убедитесь, что метрики ≥ предыдущих.
3. Скопируйте артефакты на сервер (scp/rsync).
4. Перезапустите сервис:
   - systemd: `sudo systemctl restart ml-web.service`
   - Docker: `docker compose up --build -d`
5. Убедитесь, что `/model` и `/reports/metrics` отражают обновление.

## 14. Полная последовательность действий
1. Подготовьте сервер (раздел 1).
2. Клонируйте репозиторий и создайте venv (разделы 2–3).
3. `make install` — зависимости.
4. (Опционально) `make eda` — анализ данных.
5. `make train` и/или `make train-transformer` — обучение.
6. `make evaluate` — проверка качества.
7. `make test` — убедиться в корректности вспомогательной логики.
8. `make serve` — запуск API + UI (или `docker compose up`).
9. Используйте `/predict`, `/predict_file`, панель UI.
10. Собирайте фидбэк и историю (`make feedback-export`, `make history-report`).
11. Повторяйте цикл обучения по мере поступления новых данных.

Следуя этим шагам, вы развернёте полный ML + Web стек, обеспечите контроль качества, соберёте обратную связь пользователей и сможете быстро выкатывать улучшения в продакшн.

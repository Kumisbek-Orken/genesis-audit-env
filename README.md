# GENESIS Financial Audit RL Environment

Воспроизводимый симулятор и RL-среда для обучения и оценки ИИ-аудитора на синтетических бухгалтерских данных (GENESIS Hackathon // FinTech).

Агент исследует открытые финансовые записи компании с помощью узкого набора инструментов, выявляет скрытые бухгалтерские нарушения с подтверждающими доказательствами и получает детерминированную терминальную награду на основе 1-к-1 сопоставления с закрытым эталоном. Все данные, компании, счета и суммы полностью вымышлены; проверка и запуск выполняются полностью оффлайн без доступа к сети.

## Структура репозитория

- `audit_env/` — ядро среды, генератор данных, матчер и базовые политики:
  - `schemas.py` — строгие dataclass-схемы и валидаторы (`Manifest`, `Policy`, `LedgerRow`, `Invoice`, `Payment`, `FXRate`, `TruthIssue`, `Submission`).
  - `calculator.py` — безопасный AST-калькулятор для арифметических выражений.
  - `generator.py` — каузальный генератор сбалансированной истории и инъекции 4 типов ошибок, 2 составных сценариев и законных distractors (`generate_episode`, `save_episode`, `load_episode`).
  - `matcher.py` — детерминированное жадное 1-к-1 сопоставление и расчёт P / R / F1 / награды.
  - `env.py` — среда `AuditEnv` с контрактом `reset()` / `step()`, лимитом 100 шагов, пагинацией `<= 50` и изоляцией эталона.
  - `policies.py` — 4 базовые политики (`RandomPolicy`, `WeakHeuristicPolicy`, `StrongBaselinePolicy`, `OraclePolicy`) и единый драйвер `run_episode`.
- `scripts/` — воспроизводимые CLI-скрипты:
  - `generate_data.py` — детерминированная генерация наборов эпизодов (включая split `train` / `dev`).
  - `run_eval.py` — прогон всех 4 политик на сохранённых эпизодах, печать таблицы метрик и экспорт `trajectories.jsonl`.
- `tests/` — автотесты контракта среды, схем, генератора, инвариантности, матчера, политик и CLI (174 теста).
- `DESIGN.md` — архитектурная записка (POMDP-постановка, каузальный генератор, защита от взлома награды, разбор метрик и ограничений).
- `Dockerfile` — автономный оффлайн-образ на базе `python:3.11-slim`.

## Локальный запуск

Установка зависимостей (Python 3.11+, достаточно `pytest`):

```bash
pip install -r requirements.txt
```

Запуск полного набора тестов (174 теста):

```bash
python3 -m pytest -v --tb=short
```

Генерация 10 открытых `train`- и 10 `dev`-эпизодов (по SPEC §09):

```bash
python3 scripts/generate_data.py --out-dir data --num-episodes 10 --seed 100 --splits train dev
```

Оценка всех 4 политик на `dev`-наборе и запись траекторий в `trajectories.jsonl`:

```bash
python3 scripts/run_eval.py --episodes-dir data/dev --out trajectories.jsonl
```

## Запуск в Docker (без доступа к сети)

Сборка образа:

```bash
docker build -t genesis-audit-env .
```

Оффлайн-запуск тестов (`pytest -q`) и демонстрационной оценки 4 политик:

```bash
docker run --rm --network none genesis-audit-env
```

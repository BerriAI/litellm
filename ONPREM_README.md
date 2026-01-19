# LiteLLM On-Prem Enterprise Edition

Эта версия LiteLLM включает все enterprise функции без необходимости лицензирования.

## Особенности

- Все enterprise функции разблокированы по умолчанию
- Нет ограничений на использование
- Нет запросов лицензии
- Нет предупреждений о премиум функциях
- Полная функциональность MCP, Vector Store, Guardrails и других enterprise возможностей

## Сборка

Для сборки on-prem версии используйте предоставленный скрипт:

```bash
./build_onprem.sh
```

Этот скрипт создаст Docker образ `litellm-onprem` с полностью разблокированными enterprise функциями.

## Запуск

После сборки запустите контейнер следующей командой:

```bash
docker run -p 4000:4000 litellm-onprem
```

Сервер будет доступен по адресу http://localhost:4000

## Тестирование

Вы можете протестировать работу enterprise функций с помощью скрипта:

```bash
./test_in_docker.sh
```

## Технические детали

On-prem версия собирается из локального исходного кода с включением всех enterprise модулей:

1. Установка LiteLLM в режиме редактирования (`pip install -e .`) для включения локального кода
2. Автоматический импорт enterprise модулей при запуске через изменение `__init__.py`
3. Отсутствие зависимости от PyPI версий, которые не включают enterprise функции
4. Все enterprise функции доступны без каких-либо ограничений

## Поддерживаемые функции

- Model Control Protocol (MCP)
- Vector Store
- Guardrails
- Advanced caching
- Расширенные интеграции
- Все остальные enterprise функции

## Архитектура

On-prem версия использует следующую архитектуру:

```
/app/litellm-source/
├── litellm/           # Основной пакет LiteLLM
├── enterprise/        # Enterprise модули
│   └── litellm_enterprise/
└── docker/build_from_pip/Dockerfile.onprem  # Dockerfile для on-prem сборки
```

Enterprise модули автоматически добавляются в путь Python при запуске, что позволяет использовать все enterprise функции без дополнительной настройки.
# Очередь запросов при достижении лимита параллелизма модели

## Обзор

Добавлена настройка `general_settings.model_concurrency`, которая ограничивает количество одновременно выполняемых запросов для группы моделей. Когда лимит достигнут, новый запрос ожидает освобождения слота вместо немедленного отклонения.

## Конфигурация

```yaml
general_settings:
  model_concurrency:
    gpt-4o:
      limit: 5                    # макс. параллельных запросов для этой группы моделей
      queue_timeout_seconds: 60   # время ожидания слота (по умолчанию 60)
```

## Ключевые решения

- **Ключ лимита** — группа модели (то, что клиент передаёт как `model`), общий для всех пользователей/ключей
- **Хранение слотов** — в Redis-гауге v3-лимитера, поэтому все воркеры/реплики прокси делят один счётчик
- **Таймаут** — если слот не освободился за `queue_timeout_seconds`, возвращается 429 с `RateLimitType.CONCURRENT_REQUESTS`
- **Сессия** — один HTTP-запрос занимает один слот на всё время жизни, включая стриминг
- **Интервал опроса** — 0.1 секунды (`MODEL_CONCURRENCY_QUEUE_POLL_INTERVAL_SECONDS`)

## Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `litellm/proxy/hooks/model_concurrency_queue.py` | Новый модуль: модель конфигурации и парсер |
| `litellm/proxy/hooks/parallel_request_limiter_v3.py` | Интеграция очереди: acquire/release слотов |
| `tests/unit/proxy/hooks/test_model_concurrency_queue.py` | Новые тесты поведения |

## Тесты

- 564 теста в v3-лимитере + batch post-call + legacy-лимитер + новая очередь
- Ruff strict gate: OK (правила не выросли относительно merge-base)
- Type discipline gate: OK (LIT-правила не выросли относительно merge-base)
- Live e2e: последовательные запросы проходят, параллельные встают в очередь, 429 при таймауте во время стриминга

## Пример работы

```
sequential first  (200, 0.68, "hello")      # первый запрос прошёл
sequential second (200, 0.53, "hello")      # второй прошёл после освобождения слота
concurrent fast   (200, 0.53, "hello")      # один из параллельных занял слот
concurrent queued (200, 1.15, "hello")      # второй ждал ~0.6с и прошёл
while streaming   (429, 1.02, "...")        # во время стриминга слот занят → таймаут
stream            (200, 2.04, "6 chunks")   # стриминг завершился
after stream      (200, 0.53, "hello")      # слот освобождён, запрос прошёл
```

## Ограничения

1. **Legacy v1-лимитер** (`LEGACY_MULTI_INSTANCE_RATE_LIMITING=true`) не поддерживает эту функцию
2. **Нет лимита глубины очереди** — каждый ожидающий запрос опрашивает каждые 0.1с до таймаута
3. **Поле не в UI** — `model_concurrency` не добавлено в `ConfigGeneralSettings.model_fields`, поэтому UI управления БД не будет его автодополнять

## Детали реализации

### Acquire (занятие слота)

1. В `async_pre_call_hook` вызывается `_await_model_concurrency_slot`
2. Резолвится группа модели через `_model_group_resolver`
3. В цикле с интервалом 0.1с вызывается `parallel_acquire` в Redis
4. При успехе слот сохраняется в `stash.model_concurrency_slot`
5. При таймауте выбрасывается `ProxyRateLimitError` с 429

### Release (освобождение слота)

Слот освобождается через `_release_stashed_model_concurrency_slot` во всех существующих путях:

- `async_post_call_success_hook` — успешный ответ
- `async_post_call_failure_hook` — ошибка провайдера
- `async_release_max_parallel_requests_on_disconnect` — обрыв соединения
- `async_post_call_streaming_iterator_hook` — конец стриминга

### Отказоустойчивость

- При недоступности Redis деградирует к in-memory счётчику (по умолчанию)
- При `fail_closed_rate_limit_enforcement: true` — возвращает 503 вместо деградации
- TTL слота: 1 час (`PARALLEL_REQUEST_SLOT_TTL_SECONDS = 3600`), утечки самоисправляются

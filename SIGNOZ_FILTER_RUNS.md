# Как фильтровать данные последнего запуска в SigNoz

При запуске тестов каждому запуску автоматически присваивается уникальный `run.id`, который выводится в логах:

```
======================================================================
IMPALA Full Integration Test: Actors + Learner + NATS
======================================================================
🆔 Run ID: 1709129345_a3f8d912
   Use this ID to filter data in SigNoz
======================================================================
```

## Способ 1: Фильтр по run.id (Рекомендуется)

### В Traces (http://localhost:3301/traces):
1. Откройте **Traces** → **Explorer**
2. В фильтрах добавьте: `resource.run.id = 1709129345_a3f8d912`
3. Все трейсы будут отфильтрованы по этому запуску

### В Metrics (http://localhost:3301/metrics):
1. Откройте **Metrics** → **Query Builder**
2. В секции **Where** добавьте фильтр:
   - Attribute: `resource_run_id`
   - Operator: `=`
   - Value: `1709129345_a3f8d912`

### В ClickHouse запросах:
```sql
SELECT 
    toDateTime(unix_milli/1000) as time,
    value,
    resource_attrs['run.id'] as run_id
FROM distributed_samples_v4  
WHERE metric_name = 'learner.loss'
  AND resource_attrs['run.id'] = '1709129345_a3f8d912'
ORDER BY time
```

## Способ 2: Временной фильтр (Быстрый)

В правом верхнем углу SigNoz UI:
- Выберите **"Last 5 minutes"** для недавнего запуска
- Или **"Last 15 minutes"** если тест был длиннее
- Или **Custom Range** для точного диапазона

## Способ 3: Очистка старых данных

Если тестируете локально и старые данные не нужны:

```bash
# Остановить SigNoz
cd ~/signoz
docker compose -f docker/clickhouse-setup/docker-compose.yaml down

# Удалить volumes с данными
docker volume rm clickhouse-setup_signoz-data

# Запустить заново
docker compose -f docker/clickhouse-setup/docker-compose.yaml up -d
```

## Скрипт для быстрого просмотра последнего запуска

Можно сохранить run_id в файл:

```bash
# В test_impala_full_integration.py добавить:
with open('.last_run_id', 'w') as f:
    f.write(run_id)

# Потом использовать:
RUN_ID=$(cat .last_run_id)
echo "Last run ID: $RUN_ID"
```

## Dashboard с фильтром по run_id

При создании дашборда в SigNoz:
1. Создайте Variable: `$run_id`
2. В каждой панели добавьте фильтр: `resource_run_id = $run_id`
3. В верхней части дашборда будет dropdown для выбора run_id

## Пример: Просмотр метрик последнего запуска

```bash
# 1. Запустить тест
python3 test_impala_full_integration.py

# 2. Скопировать Run ID из логов (например: 1709129345_a3f8d912)

# 3. Открыть SigNoz: http://localhost:3301

# 4. В Metrics добавить фильтр:
#    resource_run_id = 1709129345_a3f8d912

# 5. Построить график learner.loss с этим фильтром
```

## Автоматический фильтр в view_metrics.py

Если хотите видеть только последний запуск в `view_metrics.py`, добавьте фильтр:

```python
# В SQL запросе добавить WHERE:
query = f"""
    SELECT 
        toDateTime(unix_milli/1000) as timestamp,
        metric_name,
        value,
        resource_attrs['run.id'] as run_id
    FROM distributed_samples_v4
    WHERE metric_name LIKE 'learner.%' OR metric_name LIKE 'actor.%'
      AND resource_attrs['run.id'] = '{last_run_id}'
    ORDER BY timestamp DESC
    LIMIT 100
"""
```

## Полезные run_id фильтры

```sql
-- Посмотреть все доступные run_id:
SELECT DISTINCT resource_attrs['run.id'] as run_id
FROM distributed_samples_v4
ORDER BY run_id DESC
LIMIT 10;

-- Количество метрик по каждому запуску:
SELECT 
    resource_attrs['run.id'] as run_id,
    count(*) as metric_count
FROM distributed_samples_v4
GROUP BY run_id
ORDER BY run_id DESC;

-- Временной диапазон каждого запуска:
SELECT 
    resource_attrs['run.id'] as run_id,
    min(toDateTime(unix_milli/1000)) as start_time,
    max(toDateTime(unix_milli/1000)) as end_time,
    count(*) as metrics_count
FROM distributed_samples_v4
GROUP BY run_id
ORDER BY start_time DESC;
```

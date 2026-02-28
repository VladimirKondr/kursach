# 🚀 IMPALA Ultimate Dashboard Guide

## Что включено в дашборд

### 📊 Основные панели (8 панелей)

1. **🎯 Training Loss** - Главная метрика обучения
   - Должна снижаться со временем
   - Если растет или стагнирует → проблемы с обучением

2. **🎁 Rewards (Mean/Min/Max by Actor)** - Награды по акторам
   - Mean reward должен расти (модель учится генерировать лучшие молекулы)
   - Видно все акторы отдельно + min/max границы

3. **✅ SMILES Validity Ratio** - Процент валидных молекул
   - Должен быть > 95%
   - Если падает → модель генерирует невалидные структуры

4. **⚖️ Importance Weights (Aggregate)** - Агрегированные веса
   - Mean ≈ 1.0 (нормализованные)
   - **Min вес < 0.1 → какой-то актор отстал!**
   - Max вес >> 1 → сильная off-policy коррекция

5. **⚖️ Importance Weights Per Actor** - Веса каждого актора
   - Видно какой именно актор проблемный
   - Если один актор постоянно низкий → проверить model.version.lag

6. **🔄 Model Version Lag** - Отставание модели у акторов
   - Lag < 5 версий - OK
   - Lag > 10 версий - актор слишком медленный или проблемы с NATS

7. **📦 NATS Queue Depth** - Глубина очереди траекторий
   - Низкая (< 100) - OK, learner успевает
   - Высокая (> 500) - bottleneck на learner, акторы слишком быстрые

8. **⏱️ Operation Durations** - Время операций
   - Learner Update - сколько занимает обновление модели
   - Actor Trajectory Gen - полное время генерации
   - Actor Sampling - время на sampling SMILES
   - Actor Scoring - время на скоринг молекул

---

## 🔧 Как импортировать дашборд в SigNoz

### Способ 1: Через UI (если поддерживается)

1. Откройте SigNoz UI: http://localhost:3301
2. Перейдите в **Dashboards** → **+ New Dashboard**
3. Нажмите **Import** (если есть кнопка)
4. Загрузите файл `signoz_impala_dashboard.json`

### Способ 2: Создать вручную (рекомендуется)

К сожалению, SigNoz v0.113.0 не поддерживает прямой импорт JSON дашбордов через UI.
Нужно создавать панели вручную, используя SQL запросы из JSON файла.

#### Пошаговая инструкция:

1. **Создайте новый Dashboard:**
   ```
   SigNoz UI → Dashboards → + New Dashboard
   Название: "IMPALA Training Dashboard"
   ```

2. **Добавьте панель "Training Loss":**
   - Нажмите **+ Add Panel**
   - Panel Type: **Time Series**
   - Title: `🎯 Training Loss`
   - Query Builder → **ClickHouse Query**
   - SQL:
     ```sql
     SELECT 
       toDateTime(unix_milli/1000) as time, 
       value as loss 
     FROM distributed_samples_v4 
     WHERE metric_name='learner.loss' 
       AND time >= now() - INTERVAL 1 HOUR 
     ORDER BY time
     ```
   - Save Panel

3. **Добавьте панель "Rewards":**
   - **+ Add Panel** → Time Series
   - Title: `🎁 Rewards (Mean/Min/Max by Actor)`
   - Создайте **3 запроса** (Query A, B, C):
   
   **Query A - Mean Reward:**
   ```sql
   SELECT 
     toDateTime(unix_milli/1000) as time, 
     value as reward,
     attributes['actor_id'] as actor_id
   FROM distributed_samples_v4 
   WHERE metric_name='actor.reward.mean' 
     AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   ```
   
   **Query B - Max Reward:**
   ```sql
   SELECT 
     toDateTime(unix_milli/1000) as time, 
     value as reward
   FROM distributed_samples_v4 
   WHERE metric_name='actor.reward.max' 
     AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   ```
   
   **Query C - Min Reward:**
   ```sql
   SELECT 
     toDateTime(unix_milli/1000) as time, 
     value as reward
   FROM distributed_samples_v4 
   WHERE metric_name='actor.reward.min' 
     AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   ```

4. **Добавьте панель "Validity Ratio":**
   - Title: `✅ SMILES Validity Ratio`
   - SQL:
     ```sql
     SELECT 
       toDateTime(unix_milli/1000) as time, 
       value * 100 as validity_percent,
       attributes['actor_id'] as actor_id
     FROM distributed_samples_v4 
     WHERE metric_name='actor.smiles.valid.ratio' 
       AND time >= now() - INTERVAL 1 HOUR 
     ORDER BY time
     ```

5. **Добавьте панель "Importance Weights Aggregate":**
   - Title: `⚖️ Importance Weights (Aggregate)`
   - 3 запроса:
   
   ```sql
   -- Mean
   SELECT toDateTime(unix_milli/1000) as time, value as weight 
   FROM distributed_samples_v4 
   WHERE metric_name='learner.importance_weights.mean' 
     AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   
   -- Min (⚠️ КРИТИЧНО!)
   SELECT toDateTime(unix_milli/1000) as time, value as weight 
   FROM distributed_samples_v4 
   WHERE metric_name='learner.importance_weights.min' 
     AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   
   -- Max
   SELECT toDateTime(unix_milli/1000) as time, value as weight 
   FROM distributed_samples_v4 
   WHERE metric_name='learner.importance_weights.max' 
     AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   ```

6. **Добавьте панель "Per-Actor Importance Weights":**
   - Title: `⚖️ Importance Weights Per Actor`
   - SQL:
     ```sql
     SELECT 
       toDateTime(unix_milli/1000) as time, 
       value as weight,
       attributes['actor_id'] as actor_id
     FROM distributed_samples_v4 
     WHERE metric_name='learner.importance_weight' 
       AND time >= now() - INTERVAL 1 HOUR 
     ORDER BY time
     ```

7. **Добавьте панель "Model Version Lag":**
   - Title: `🔄 Model Version Lag`
   - SQL:
     ```sql
     SELECT 
       toDateTime(unix_milli/1000) as time, 
       value as lag,
       attributes['actor_id'] as actor_id
     FROM distributed_samples_v4 
     WHERE metric_name='model.version.lag' 
       AND time >= now() - INTERVAL 1 HOUR 
     ORDER BY time
     ```

8. **Добавьте панель "Queue Depth":**
   - Title: `📦 NATS Queue Depth`
   - SQL:
     ```sql
     SELECT 
       toDateTime(unix_milli/1000) as time, 
       value as depth
     FROM distributed_samples_v4 
     WHERE metric_name='nats.queue.depth' 
       AND time >= now() - INTERVAL 1 HOUR 
     ORDER BY time
     ```

9. **Добавьте панель "Operation Durations":**
   - Title: `⏱️ Operation Durations`
   - 4 запроса:
   
   ```sql
   -- Learner Update
   SELECT toDateTime(unix_milli/1000) as time, value as duration 
   FROM distributed_samples_v4 
   WHERE metric_name='learner.update.duration' AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   
   -- Actor Trajectory Generation
   SELECT toDateTime(unix_milli/1000) as time, value as duration 
   FROM distributed_samples_v4 
   WHERE metric_name='actor.trajectory_generation.duration' AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   
   -- Actor Sampling
   SELECT toDateTime(unix_milli/1000) as time, value as duration 
   FROM distributed_samples_v4 
   WHERE metric_name='actor.sampling.duration' AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   
   -- Actor Scoring  
   SELECT toDateTime(unix_milli/1000) as time, value as duration 
   FROM distributed_samples_v4 
   WHERE metric_name='actor.scoring.duration' AND time >= now() - INTERVAL 1 HOUR 
   ORDER BY time
   ```

10. **Сохраните Dashboard:**
    - Нажмите **Save Dashboard**
    - Настройте auto-refresh: **5 seconds** или **10 seconds**

---

## 📈 Как интерпретировать метрики

### ✅ Признаки здорового обучения:

- **Loss снижается** (может иметь флуктуации, но тренд вниз)
- **Reward растет** (модель учится генерировать лучшие молекулы)
- **Validity > 95%** (модель не разучилась генерировать валидные структуры)
- **Importance weights mean ≈ 1.0** (off-policy коррекция работает)
- **Importance weights min > 0.1** (все акторы актуальны)
- **Model version lag < 5** (акторы быстро получают обновления)
- **Queue depth < 200** (learner успевает обрабатывать)

### ⚠️ Проблемные паттерны:

1. **Loss растет или стагнирует:**
   - Learning rate слишком высокий → уменьшить
   - Проблемы с V-trace коррекцией → проверить importance weights
   - Плохая scoring function → молекулы не имеют хорошего gradient

2. **Importance weights min < 0.05:**
   - Какой-то актор сильно отстал от learner
   - Проверить model.version.lag для этого актора
   - Возможно, нужно чаще обновлять модели на акторах

3. **Validity падает < 90%:**
   - Модель разучивается генерировать валидные молекулы
   - Слишком агрессивное обучение → уменьшить learning rate
   - Нужно больше regularization (увеличить prior weight)

4. **Queue depth > 500 и растет:**
   - Learner bottleneck (медленнее акторов)
   - Решения:
     - Уменьшить количество акторов
     - Увеличить batch_size на learner
     - Оптимизировать learner.update (GPU?)

5. **Model version lag > 10:**
   - Актор не получает обновления
   - Проблемы с NATS communication
   - Проверить логи актора и Swarm

---

## 🔍 Дополнительные полезные запросы

### Посмотреть все доступные метрики:
```sql
SELECT DISTINCT metric_name 
FROM distributed_samples_v4 
WHERE time >= now() - INTERVAL 1 HOUR 
ORDER BY metric_name
```

### Статистика по ошибкам:
```sql
SELECT 
  metric_name,
  count() as error_count
FROM distributed_samples_v4 
WHERE metric_name LIKE '%error%' 
  AND time >= now() - INTERVAL 1 HOUR 
GROUP BY metric_name
ORDER BY error_count DESC
```

### Throughput акторов (траекторий в секунду):
```sql
SELECT 
  attributes['actor_id'] as actor_id,
  sum(value) / 3600 as trajectories_per_second
FROM distributed_samples_v4 
WHERE metric_name = 'actor.trajectories_generated.total'
  AND time >= now() - INTERVAL 1 HOUR 
GROUP BY actor_id
ORDER BY actor_id
```

### Средний размер модели:
```sql
SELECT 
  avg(value) / 1024 / 1024 as avg_model_size_mb
FROM distributed_samples_v4 
WHERE metric_name = 'model.size_bytes'
  AND time >= now() - INTERVAL 1 HOUR
```

---

## 🎯 Быстрый чеклист перед запуском теста

1. ✅ SigNoz запущен: `docker ps | grep signoz`
2. ✅ NATS очищен: порт 4222 свободен
3. ✅ Dashboard создан в SigNoz UI
4. ✅ Временной фильтр установлен: "Last 1 hour" или больше

---

## 🚀 Запуск теста и мониторинг

```bash
# Очистить старые метрики (опционально)
docker exec signoz-clickhouse clickhouse-client --query "DELETE FROM distributed_samples_v4 WHERE metric_name LIKE 'learner.%' OR metric_name LIKE 'actor.%'"

# Запустить тест
python3 test_impala_full_integration.py

# Открыть дашборд
# http://localhost:3301/dashboard
```

**Мониторинг в реальном времени:**
- Откройте дашборд в браузере
- Установите auto-refresh: 5-10 секунд
- Наблюдайте за метриками во время обучения

---

## 📚 Дополнительные возможности

### Traces (распределенная трассировка)

Помимо метрик, у нас также собираются traces! Они показывают:
- Полный путь траектории от актора до learner
- Время на каждом этапе (sampling, scoring, NATS, V-trace)
- Span relationships (родитель-потомок)

**Посмотреть traces:**
```
SigNoz UI → Traces
Фильтры:
- service.name = "impala-learner" или "impala-actor"  
- operation = "learner.update" или "actor.generate_trajectories"
```

### Alerts (алерты)

Можно настроить алерты в SigNoz:

1. **Loss не снижается:**
   ```sql
   -- Alert если loss не уменьшился за 10 минут
   SELECT avg(value) as avg_loss
   FROM distributed_samples_v4 
   WHERE metric_name='learner.loss' 
     AND time >= now() - INTERVAL 10 MINUTE
   HAVING avg_loss > 
     (SELECT avg(value) FROM distributed_samples_v4 
      WHERE metric_name='learner.loss' 
        AND time BETWEEN now() - INTERVAL 20 MINUTE AND now() - INTERVAL 10 MINUTE)
   ```

2. **Validity упала:**
   ```sql
   SELECT avg(value) as avg_validity
   FROM distributed_samples_v4 
   WHERE metric_name='actor.smiles.valid.ratio' 
     AND time >= now() - INTERVAL 5 MINUTE
   HAVING avg_validity < 0.90  -- Alert если < 90%
   ```

3. **Queue depth слишком высокий:**
   ```sql
   SELECT max(value) as max_depth
   FROM distributed_samples_v4 
   WHERE metric_name='nats.queue.depth' 
     AND time >= now() - INTERVAL 5 MINUTE
   HAVING max_depth > 500
   ```

---

## 💡 Советы по оптимизации

1. **Если loss скачет сильно:**
   - Уменьшить learning rate
   - Увеличить batch size на learner
   - Проверить clip_rho в importance weights

2. **Если акторы отстают (high lag):**
   - Увеличить задержку на акторах (ACTOR_DELAY)
   - Уменьшить количество акторов
   - Оптимизировать model serialization/deserialization

3. **Если validity падает:**
   - Увеличить prior_weight (больше регуляризации)
   - Уменьшить learning rate
   - Проверить что prior модель заморожена

4. **Если queue растет:**
   - Learner слишком медленный → GPU?
   - Или уменьшить NUM_ACTORS
   - Или увеличить BATCH_SIZE на learner

---

**Готово!** 🎉 Теперь у вас полный контроль над обучением IMPALA!

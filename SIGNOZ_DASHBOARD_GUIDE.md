# 📊 Создание Dashboard IMPALA в SigNoz - Полная Инструкция

## ✅ РАБОЧЕЕ РЕШЕНИЕ

Проблема была в том, что OpenTelemetry Histogram хранит **кумулятивные** значения (sum и count всегда растут). Чтобы получить реальный loss на каждом шаге, нужно вычислить **дельты** между соседними точками.

---

## 🎯 Пошаговая Инструкция

### 1️⃣ Откройте SigNoz

```
http://localhost:8080
```

### 2️⃣ Создайте Dashboard

1. Левое меню → **Dashboards**
2. Нажмите **"+ New Dashboard"**
3. Название: `IMPALA Training Monitor`
4. Description: `Real-time monitoring of IMPALA RL training`
5. Нажмите **"Save"**

### 3️⃣ Добавьте Panel #1 - Loss

1. Нажмите **"+ Add Panel"** внутри dashboard
2. Вверху переключитесь на вкладку **"ClickHouse Query"** (НЕ Query Builder!)
3. Вставьте этот SQL запрос:

```sql
SELECT 
    time,
    avg_loss
FROM (
    SELECT 
        toDateTime(unix_milli / 1000) as time,
        round(
            (argMaxIf(value, unix_milli, metric_name = 'learner.loss.sum') - 
             lagInFrame(argMaxIf(value, unix_milli, metric_name = 'learner.loss.sum'), 1, 0) OVER (ORDER BY unix_milli)) 
            / 
            nullIf(argMaxIf(value, unix_milli, metric_name = 'learner.loss.count') - 
                   lagInFrame(argMaxIf(value, unix_milli, metric_name = 'learner.loss.count'), 1, 0) OVER (ORDER BY unix_milli), 0),
            2
        ) as avg_loss
    FROM signoz_metrics.samples_v4
    WHERE metric_name IN ('learner.loss.sum', 'learner.loss.count')
    AND unix_milli >= toUnixTimestamp(now() - INTERVAL 1 HOUR) * 1000
    GROUP BY unix_milli
)
WHERE avg_loss IS NOT NULL AND avg_loss > 0
ORDER BY time
```

4. Нажмите **"Stage & Run Query"** (синяя кнопка справа сверху)
5. Вы должны увидеть данные в Preview!

### 4️⃣ Настройте Visualization для Loss

**Panel Options (справа):**
```
Title: Learner Loss (Average per Update)
Description: Real average loss computed from deltas
```

**Visualization:**
```
Panel Type: Time Series
Graph Type: Line
Line Width: 2
Fill Opacity: 0.1
Point Size: 4
```

**Axes:**
```
Left Y Axis:
  - Label: "Loss"
  - Scale: Linear
  - Show Grid: Yes
```

**Legend:**
```
Show Legend: Yes
Position: Bottom
Display Mode: List
Values: Current, Min, Max, Mean
```

5. Нажмите **"Apply"** (правый верхний угол)

### 5️⃣ Добавьте Panel #2 - Reward

1. В dashboard нажмите **"+ Add Panel"** снова
2. Переключитесь на **"ClickHouse Query"**
3. Вставьте:

```sql
SELECT 
    time,
    avg_reward
FROM (
    SELECT 
        toDateTime(unix_milli / 1000) as time,
        round(
            (argMaxIf(value, unix_milli, metric_name = 'actor.reward.mean.sum') - 
             lagInFrame(argMaxIf(value, unix_milli, metric_name = 'actor.reward.mean.sum'), 1, 0) OVER (ORDER BY unix_milli)) 
            / 
            nullIf(argMaxIf(value, unix_milli, metric_name = 'actor.reward.mean.count') - 
                   lagInFrame(argMaxIf(value, unix_milli, metric_name = 'actor.reward.mean.count'), 1, 0) OVER (ORDER BY unix_milli), 0),
            4
        ) as avg_reward
    FROM signoz_metrics.samples_v4
    WHERE metric_name IN ('actor.reward.mean.sum', 'actor.reward.mean.count')
    AND unix_milli >= toUnixTimestamp(now() - INTERVAL 1 HOUR) * 1000
    GROUP BY unix_milli
)
WHERE avg_reward IS NOT NULL
ORDER BY time
```

4. **Title:** `Actor Mean Reward`
5. **Y-Axis Label:** `Reward`
6. Нажмите **"Apply"**

### 6️⃣ Добавьте Panel #3 - Validity

1. **"+ Add Panel"** → **"ClickHouse Query"**
2. Вставьте:

```sql
SELECT 
    time,
    validity
FROM (
    SELECT 
        toDateTime(unix_milli / 1000) as time,
        round(
            (argMaxIf(value, unix_milli, metric_name = 'actor.smiles.valid.ratio.sum') - 
             lagInFrame(argMaxIf(value, unix_milli, metric_name = 'actor.smiles.valid.ratio.sum'), 1, 0) OVER (ORDER BY unix_milli)) 
            / 
            nullIf(argMaxIf(value, unix_milli, metric_name = 'actor.smiles.valid.ratio.count') - 
                   lagInFrame(argMaxIf(value, unix_milli, metric_name = 'actor.smiles.valid.ratio.count'), 1, 0) OVER (ORDER BY unix_milli), 0),
            4
        ) as validity
    FROM signoz_metrics.samples_v4
    WHERE metric_name IN ('actor.smiles.valid.ratio.sum', 'actor.smiles.valid.ratio.count')
    AND unix_milli >= toUnixTimestamp(now() - INTERVAL 1 HOUR) * 1000
    GROUP BY unix_milli
)
WHERE validity IS NOT NULL AND validity >= 0 AND validity <= 1
ORDER BY time
```

3. **Title:** `SMILES Validity Ratio`
4. **Y-Axis Label:** `Validity (0-1)`
5. **Y-Axis Min:** `0`, **Max:** `1`
6. Нажмите **"Apply"**

### 7️⃣ Настройте Dashboard Layout

1. Перетащите панели в удобное расположение:
   ```
   ┌────────────────────┬────────────────────┐
   │  Loss              │  Reward            │
   │                    │                    │
   ├────────────────────┴────────────────────┤
   │  Validity                               │
   │                                         │
   └─────────────────────────────────────────┘
   ```

2. Настройте Auto-Refresh:
   - Правый верхний угол → выберите refresh rate: **10s** или **30s**

3. Time Range:
   - Правый верхний угол → выберите **Last 1 hour**

4. Нажмите **"Save Dashboard"**

---

## 🔍 Как это работает (объяснение SQL)

### Проблема:
В ClickHouse метрики хранятся как кумулятивные:
```
time        | learner.loss.sum | learner.loss.count
18:42:04    | 7301             | 1
18:42:34    | 24048            | 3   ← сумма всех loss до этого момента
18:43:04    | 32189            | 5   ← продолжает расти
```

### Решение:
Вычислить дельты (изменения) между соседними точками:
```sql
lagInFrame(..., 1, 0) OVER (ORDER BY unix_milli)
```

Это берёт **предыдущее значение** (lag) и вычитает:
```
delta_sum = current_sum - previous_sum
delta_count = current_count - previous_count
avg_loss = delta_sum / delta_count
```

Пример:
```
Точка 2: sum=24048, count=3
Точка 1: sum=7301,  count=1

delta_sum = 24048 - 7301 = 16747
delta_count = 3 - 1 = 2
avg_loss = 16747 / 2 = 8373.5  ← реальный средний loss на этом шаге!
```

---

## 📈 Ожидаемый Результат

### Loss Panel:
Вы увидите график который **может падать** (улучшение):
```
Loss
8000 │     ●
     │      ╲     ●
7000 │       ●   ╱
     │        ╲ ╱
6000 │         ●
     │          ╲
5000 │           ●
     │            ╲
4000 │             ●
     │              ╲
3000 │               ●
     │                ╲
1000 │                 ●  ← минимум, модель учится!
     └──────────────────────
```

### Reward Panel:
Reward должен **расти** при успешном обучении:
```
Reward
0.8 │                    ●
    │                  ╱
0.7 │                ●
    │              ╱
0.6 │            ●
    │          ╱
0.5 │        ●
    └──────────────────────
```

### Validity Panel:
Должен быть стабильно высоким (>0.9):
```
Validity
1.0 │  ●━━●━━●━━●━━●━━●
    │
0.9 │
    │
0.8 │
    └──────────────────────
```

---

## 🎨 Дополнительные Улучшения

### Добавить Alert на низкий Validity:

1. В Validity panel → **Alert** tab
2. Condition: `validity < 0.7`
3. Alert name: `Low SMILES Validity`
4. Notification channel: Email/Slack

### Добавить Variables для гибкости:

1. Dashboard Settings → **Variables** → **+ Add Variable**
2. Name: `time_range`
3. Type: `Interval`
4. Options: `30m, 1h, 3h, 6h, 12h, 24h`
5. В запросах замените `1 HOUR` на `$time_range`

---

## ✅ Проверка Работоспособности

### Тест 1: Запустите training
```bash
python3 test_impala_full_integration.py
```

### Тест 2: Откройте Dashboard
- Через 30-60 секунд вы должны увидеть новые точки на графиках
- Loss должен меняться (не монотонно расти!)
- Validity должен быть >0.9

### Тест 3: Проверьте данные напрямую
```bash
python3 view_metrics.py
```

Убедитесь что значения совпадают с dashboard.

---

## 🐛 Troubleshooting

### "No Data" в панелях:

1. Проверьте что метрики есть:
   ```bash
   docker exec signoz-clickhouse clickhouse-client --query \
     "SELECT count(*) FROM signoz_metrics.samples_v4 \
      WHERE metric_name LIKE 'learner.loss%'"
   ```

2. Если 0 - запустите тест: `python3 test_impala_full_integration.py`

3. Проверьте OTEL Collector:
   ```bash
   docker logs signoz-otel-collector --tail 50
   ```

### SQL Errors:

1. Убедитесь что используете **ClickHouse Query** (не Query Builder)
2. Проверьте что ваша версия SigNoz поддерживает `lagInFrame`
3. Попробуйте упрощённый запрос (без window functions):
   ```sql
   SELECT 
       toDateTime(unix_milli / 1000) as time,
       round(sum(value) / count(), 2) as avg_loss
   FROM signoz_metrics.samples_v4
   WHERE metric_name = 'learner.loss.sum'
   AND unix_milli >= toUnixTimestamp(now() - INTERVAL 1 HOUR) * 1000
   GROUP BY time
   ORDER BY time
   ```

### График всё равно растёт:

1. Убедитесь что в запросе используется **`lagInFrame`** (для дельт)
2. Проверьте что условие `WHERE avg_loss > 0` не фильтрует данные
3. Убедитесь что не используете Query Builder (только ClickHouse Query!)

---

## 💡 Итого

✅ Используйте **ClickHouse Query** (не Query Builder)  
✅ SQL запрос вычисляет **дельты** между точками  
✅ График показывает **реальный средний loss** на каждом шаге  
✅ Loss **может падать** - это признак обучения!  

Готовый dashboard теперь показывает правильные метрики! 🎉

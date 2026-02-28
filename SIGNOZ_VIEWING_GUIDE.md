# 📊 Руководство по просмотру метрик IMPALA в SigNoz

## 🎯 Что искать в интерфейсе

### Обзор
Вы видите **200+ метрик**, но **только ~20-25 относятся к IMPALA**. Остальное - системные метрики (CPU, память, Docker, OTEL, и т.д.).

---

## 🔍 Как найти метрики IMPALA

### Метод 1: Поиск по префиксу (РЕКОМЕНДУЕТСЯ)

В SigNoz используйте поиск по этим префиксам:

```
learner.     ← Метрики обучения (САМОЕ ВАЖНОЕ!)
actor.       ← Метрики генерации траекторий
```

---

## 📈 Ключевые метрики для проверки обучения

### 🎓 ОБУЧЕНИЕ (Learner) - ГЛАВНОЕ!

#### 1. **learner.loss** 
- **ЧТО СМОТРЕТЬ:** График должен идти **ВНИЗ** ↘️
- **ЗНАЧЕНИЕ:** Если loss уменьшается → модель учится! ✅
- **ПРОБЛЕМА:** Если loss растёт или не меняется → проблемы ❌

#### 2. **learner.agent_nll.mean** (Agent Negative Log Likelihood)
- **ЧТО СМОТРЕТЬ:** Средняя отрицательная log вероятность действий агента
- **НОРМА:** Со временем должна уменьшаться (агент становится увереннее)

#### 3. **learner.prior_nll.mean** (Prior Negative Log Likelihood)
- **ЧТО СМОТРЕТЬ:** Средняя отрицательная log вероятность prior модели
- **СРАВНЕНИЕ:** Сравните с agent_nll - агент должен быть лучше prior

#### 4. **learner.importance_weights.mean/max**
- **ЧТО СМОТРЕТЬ:** Веса V-trace для коррекции off-policy обучения
- **НОРМА:** Близко к 1.0 = хорошо, очень большие значения = проблемы

#### 5. **learner.update.duration**
- **ЧТО СМОТРЕТЬ:** Время одного обновления модели
- **ПРОБЛЕМА:** Если слишком долго (>5-10 сек) → батч слишком большой

#### 6. **learner.trajectories_fetched.total** (counter)
- **ЧТО СМОТРЕТЬ:** Сколько траекторий получил learner
- **НОРМА:** Должно расти линейно, без больших пауз

---

### 🎬 ГЕНЕРАЦИЯ (Actor)

#### 7. **actor.reward.mean** 
- **ЧТО СМОТРЕТЬ:** Средняя награда за траектории
- **ЗНАЧЕНИЕ:** Если растёт → агент генерирует лучшие молекулы ✅

#### 8. **actor.reward.std/max/min**
- **ЧТО СМОТРЕТЬ:** Разброс наград
- **АНАЛИЗ:** Большой std = агент исследует разные решения

#### 9. **actor.smiles.valid.ratio** 
- **ЧТО СМОТРЕТЬ:** Процент валидных молекул (должен быть >0.7, лучше >0.9)
- **КРИТИЧНО:** Если <0.5 → модель генерирует мусор ❌

#### 10. **actor.trajectories_generated.total** (counter)
- **ЧТО СМОТРЕТЬ:** Сколько траекторий создали actors
- **НОРМА:** Должно расти линейно = actors работают

#### 11. **actor.trajectory_generation.duration**
- **ЧТО СМОТРЕТЬ:** Время генерации одной траектории
- **ПРОБЛЕМА:** Если слишком долго (>30 сек) → проблемы со scoring

---

### 🔧 ТЕХНИЧЕСКИЕ (Опционально)

#### 12. **learner.batch.size**
- Размер батча траекторий для обучения

#### 13. **learner.model_commits.total** (counter)
- Сколько раз обновили модель в NATS

#### 14. **actor.model_load.duration**
- Время загрузки новой версии модели актором

---

## 🎨 Создание Dashboard в SigNoz

### Шаг 1: Фильтрация метрик
1. Откройте http://localhost:8080
2. Перейдите в **Dashboard** → **+ New Dashboard**
3. Название: "IMPALA Training Monitor"

### Шаг 2: Добавить ключевые панели

#### Панель 1: Loss (САМОЕ ВАЖНОЕ)
- **Metric:** `learner.loss`
- **Type:** Line Chart
- **Aggregation:** Mean
- **Time:** Last 15 minutes

#### Панель 2: Rewards
- **Metrics:** 
  - `actor.reward.mean` (Line)
  - `actor.reward.max` (Line)
  - `actor.reward.min` (Line)
- **Type:** Multi-line Chart

#### Панель 3: SMILES Validity
- **Metric:** `actor.smiles.valid.ratio`
- **Type:** Line Chart
- **Alert on:** <0.7 (low validity)

#### Панель 4: Траектории
- **Metrics:**
  - `actor.trajectories_generated.total` (Counter)
  - `learner.trajectories_fetched.total` (Counter)
- **Type:** Gauge или Counter
- **Purpose:** Проверить, что данные идут

#### Панель 5: Importance Weights
- **Metrics:**
  - `learner.importance_weights.mean`
  - `learner.importance_weights.max`
- **Type:** Line Chart
- **Alert on:** mean > 3.0 или max > 10.0 (слишком большие веса)

---

## 🔎 TRACES (Распределённая трассировка)

### Как посмотреть traces:

1. Перейдите в **Traces** → **APM**
2. Фильтр: `service.name = reinvent-impala` (если задан)
3. Ищите spans:
   - `actor.collect_trajectory` - полный цикл генерации траектории
   - `learner.update` - полный цикл обучения
   - `actor.sampling.duration` - время генерации SMILES
   - `actor.scoring.duration` - время подсчёта наград
   - `learner.compute_importance_weights` - вычисление V-trace весов

### Что смотреть в traces:

#### Идеальный trace "actor.collect_trajectory":
```
actor.collect_trajectory (1000ms)
  ├─ actor.sampling.duration (500ms)     ← генерация SMILES
  └─ actor.scoring.duration (500ms)      ← подсчёт наград
```

#### Идеальный trace "learner.update":
```
learner.update (3000ms)
  ├─ learner.compute_target_log_probs (1000ms)     ← forward pass
  └─ learner.compute_importance_weights (500ms)    ← V-trace веса
```

#### ПРОБЛЕМЫ в traces:
- ❌ Очень длинные spans (>10 секунд) → bottleneck
- ❌ Большой gap между spans → медленная сеть/NATS
- ❌ Много failed spans → ошибки в коде

---

## 📊 Пример реального обучения

### Хорошие признаки (модель учится):
✅ `learner.loss` постепенно падает (например, 15.0 → 12.0 → 10.0)  
✅ `actor.reward.mean` растёт или стабильный  
✅ `actor.smiles.valid.ratio` > 0.7  
✅ `learner.trajectories_fetched.total` растёт линейно без пауз  
✅ `learner.importance_weights.mean` ≈ 1.0 (не сильно отклоняется)

### Плохие признаки (проблемы):
❌ `learner.loss` растёт или NaN  
❌ `actor.reward.mean` падает  
❌ `actor.smiles.valid.ratio` < 0.5  
❌ `learner.trajectories_fetched.total` не растёт (очередь пустая)  
❌ `learner.importance_weights.max` > 100 (катастрофически большие веса)  
❌ Traces показывают failed spans

---

## 🚀 Быстрая проверка "всё ли работает"

### За 1 минуту:

1. **Откройте Dashboard → Metrics**
2. **Найдите:** `learner.loss`
3. **Проверьте:** График идёт вниз? ✅ → **РАБОТАЕТ!**
4. **Найдите:** `actor.smiles.valid.ratio`
5. **Проверьте:** Значение > 0.7? ✅ → **ГЕНЕРАЦИЯ OK!**

Если оба условия выполнены → **система работает и модель обучается!** 🎉

---

## 📝 Команды для просмотра метрик из терминала

Если SigNoz не загружается или хотите быстрый текстовый просмотр:

```bash
# Посмотреть логи теста
tail -100 <лог файл теста>

# Проверить metrics напрямую в ClickHouse (если нужно)
docker exec -it signoz-clickhouse clickhouse-client --query \
  "SELECT metric_name, avg(value) as avg_value FROM signoz_metrics.samples_v2 \
   WHERE metric_name LIKE 'learner.%' OR metric_name LIKE 'actor.%' \
   GROUP BY metric_name ORDER BY metric_name"
```

---

## 💡 Совет: Скрипт для мониторинга

Используйте `monitor_training.py` (если создан) для real-time мониторинга в терминале:

```bash
python monitor_training.py
```

Он покажет ключевые метрики в терминале без необходимости открывать браузер.

---

## ❓ FAQ

**Q: Почему я вижу 200+ метрик?**  
A: SigNoz собирает системные метрики (CPU, память, Docker, OTEL collector). Ваши IMPALA метрики имеют префиксы `learner.*` и `actor.*`.

**Q: Какая ОДНА метрика самая важная?**  
A: **`learner.loss`** - если она падает, модель учится!

**Q: Метрики не появляются в SigNoz?**  
A: Проверьте:
1. OTEL Collector работает: `docker ps | grep otel-collector`
2. Тест запущен и работает
3. Подождите 30-60 секунд (буферизация)
4. Порт 4317/4318 открыт для OTEL

**Q: Как долго ждать изменений в метриках?**  
A: Обновления каждые 10-30 секунд. Для значимых изменений (loss падает) нужно несколько минут обучения.

---

## 🎯 Итого: 3 метрики для быстрой проверки

1. **`learner.loss`** ↘️ - падает → учится ✅
2. **`actor.reward.mean`** ↗️ - растёт → улучшается ✅
3. **`actor.smiles.valid.ratio`** >0.7 - валидность OK ✅

**Всё остальное - детали!** 🚀

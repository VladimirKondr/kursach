#!/usr/bin/env python3
"""
Быстрый просмотр ключевых метрик IMPALA из SigNoz ClickHouse.
Показывает только важные метрики для проверки обучения.

Использование:
    python3 view_metrics.py                    # Все данные за последние 15 минут
    python3 view_metrics.py --last-run         # Только последний запуск теста
    python3 view_metrics.py --run-id <ID>      # Конкретный run ID
"""

import argparse
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

# Цвета для вывода
class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def query_clickhouse(query: str) -> List[str]:
    """Выполнить запрос к ClickHouse в Docker контейнере SigNoz."""
    try:
        result = subprocess.run(
            ['docker', 'exec', 'signoz-clickhouse', 'clickhouse-client', '--query', query],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [line for line in result.stdout.strip().split('\n') if line]
        else:
            print(f"{Color.RED}Ошибка запроса: {result.stderr}{Color.END}")
            return []
    except subprocess.TimeoutExpired:
        print(f"{Color.RED}Timeout при запросе к ClickHouse{Color.END}")
        return []
    except FileNotFoundError:
        print(f"{Color.RED}Docker не найден. Убедитесь, что Docker установлен и запущен.{Color.END}")
        return []
    except Exception as e:
        print(f"{Color.RED}Ошибка: {e}{Color.END}")
        return []


def get_last_run_id() -> Optional[str]:
    """Получить run_id последнего запуска теста."""
    query = """
    SELECT resource_attrs['run.id'] as run_id
    FROM distributed_samples_v4
    WHERE resource_attrs['run.id'] != ''
    ORDER BY unix_milli DESC
    LIMIT 1
    FORMAT TSV
    """
    result = query_clickhouse(query)
    if result and result[0]:
        return result[0].strip()
    return None


def get_available_run_ids(limit: int = 10) -> List[Tuple[str, str, str]]:
    """Получить список доступных run_id с временными метками."""
    query = f"""
    SELECT 
        resource_attrs['run.id'] as run_id,
        formatDateTime(min(toDateTime(unix_milli/1000)), '%Y-%m-%d %H:%M:%S') as start_time,
        formatDateTime(max(toDateTime(unix_milli/1000)), '%Y-%m-%d %H:%M:%S') as end_time
    FROM distributed_samples_v4
    WHERE resource_attrs['run.id'] != ''
    GROUP BY run_id
    ORDER BY min(unix_milli) DESC
    LIMIT {limit}
    FORMAT TSV
    """
    results = query_clickhouse(query)
    if results:
        return [tuple(line.split('\t')) for line in results if line]
    return []


def get_metric_stats(metric_name: str, time_window_minutes: int = 15, run_id: Optional[str] = None) -> Tuple[float, float, float, int]:
    """Получить статистику по метрике за последние N минут.
    
    Для histogram метрик используем sum/count для получения среднего.
    Для gauge метрик используем значение напрямую.
    """
    # Добавить фильтр по run_id если указан
    run_id_filter = ""
    if run_id:
        run_id_filter = f"AND resource_attrs['run.id'] = '{run_id}'"
    
    # Для гистограмм используем .sum и .count суффиксы
    query = f"""
    SELECT 
        round(sum(sum_val) / nullIf(sum(count_val), 0), 4) as avg_value,
        round(min(min_val), 4) as min_value,
        round(max(max_val), 4) as max_value,
        sum(count_val) as num_samples
    FROM (
        SELECT
            sumIf(value, metric_name = '{metric_name}.sum') as sum_val,
            sumIf(value, metric_name = '{metric_name}.count') as count_val,
            minIf(value, metric_name = '{metric_name}.min') as min_val,
            maxIf(value, metric_name = '{metric_name}.max') as max_val
        FROM signoz_metrics.samples_v4
        WHERE (metric_name = '{metric_name}.sum' 
            OR metric_name = '{metric_name}.count'
            OR metric_name = '{metric_name}.min'
            OR metric_name = '{metric_name}.max')
        AND unix_milli >= (toUnixTimestamp(now()) - {time_window_minutes} * 60) * 1000
        {run_id_filter}
        GROUP BY fingerprint
    )
    FORMAT TSV
    """
    
    result = query_clickhouse(query)
    if result and len(result) > 0:
        parts = result[0].split('\t')
        if len(parts) == 4:
            try:
                avg = float(parts[0]) if parts[0] not in ('nan', 'inf', '-inf', '') else 0.0
                min_val = float(parts[1]) if parts[1] not in ('nan', 'inf', '-inf', '') else 0.0
                max_val = float(parts[2]) if parts[2] not in ('nan', 'inf', '-inf', '') else 0.0
                count = int(float(parts[3])) if parts[3] else 0
                return avg, min_val, max_val, count
            except (ValueError, IndexError):
                pass
    
    return 0.0, 0.0, 0.0, 0


def get_counter_value(metric_name: str, run_id: Optional[str] = None) -> int:
    """Получить значение счётчика (counter)."""
    # Добавить фильтр по run_id если указан
    run_id_filter = ""
    if run_id:
        run_id_filter = f"AND resource_attrs['run.id'] = '{run_id}'"
    
    query = f"""
    SELECT round(max(value), 0) as total
    FROM signoz_metrics.samples_v4
    WHERE metric_name = '{metric_name}'
    AND unix_milli >= (toUnixTimestamp(now()) - 15 * 60) * 1000
    {run_id_filter}
    FORMAT TSV
    """
    
    result = query_clickhouse(query)
    if result and len(result) > 0:
        try:
            return int(float(result[0]))
        except (ValueError, IndexError):
            pass
    
    return 0


def get_all_impala_metrics() -> List[str]:
    """Получить список всех IMPALA метрик в базе (без histogram суффиксов)."""
    query = """
    SELECT DISTINCT 
        replaceRegexpOne(metric_name, '\\.(bucket|count|sum|min|max)$', '') as base_metric
    FROM signoz_metrics.samples_v4
    WHERE (metric_name LIKE 'learner.%' OR metric_name LIKE 'actor.%')
    AND metric_name NOT LIKE '%.bucket'
    ORDER BY base_metric
    FORMAT TSV
    """
    
    return query_clickhouse(query)


def print_section(title: str):
    """Напечатать заголовок секции."""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{title:^60}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}\n")


def print_metric(name: str, avg: float, min_val: float, max_val: float, samples: int, 
                 good_threshold: float = None, bad_threshold: float = None, 
                 lower_is_better: bool = False):
    """Напечатать метрику с цветовым кодированием."""
    
    # Определить цвет на основе порогов
    color = Color.END
    status = ""
    
    if samples == 0:
        color = Color.RED
        status = "⚠️ НЕТ ДАННЫХ"
    elif good_threshold is not None and bad_threshold is not None:
        if lower_is_better:
            if avg <= good_threshold:
                color = Color.GREEN
                status = "✅ ОТЛИЧНО"
            elif avg <= bad_threshold:
                color = Color.YELLOW
                status = "⚠️ ПРИЕМЛЕМО"
            else:
                color = Color.RED
                status = "❌ ПЛОХО"
        else:
            if avg >= good_threshold:
                color = Color.GREEN
                status = "✅ ОТЛИЧНО"
            elif avg >= bad_threshold:
                color = Color.YELLOW
                status = "⚠️ ПРИЕМЛЕМО"
            else:
                color = Color.RED
                status = "❌ ПЛОХО"
    
    print(f"{Color.BOLD}{name}:{Color.END}")
    print(f"  Среднее: {color}{avg:>10.4f}{Color.END}  |  Мин: {min_val:>10.4f}  |  Макс: {max_val:>10.4f}  |  Сэмплов: {samples}")
    if status:
        print(f"  Статус: {status}")
    print()


def print_counter(name: str, value: int):
    """Напечатать счётчик."""
    color = Color.GREEN if value > 0 else Color.RED
    status = "✅ РАБОТАЕТ" if value > 0 else "⚠️ НЕТ ДАННЫХ"
    
    print(f"{Color.BOLD}{name}:{Color.END}")
    print(f"  Всего: {color}{value:>10}{Color.END}  |  Статус: {status}")
    print()


def check_clickhouse_connection() -> bool:
    """Проверить соединение с ClickHouse."""
    result = subprocess.run(
        ['docker', 'ps', '--filter', 'name=signoz-clickhouse', '--format', '{{.Status}}'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    
    if result.returncode == 0 and 'Up' in result.stdout:
        return True
    return False


def main():
    """Основная функция - показать ключевые метрики IMPALA."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Просмотр IMPALA метрик из SigNoz',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                      # Все данные за последние 15 минут
  %(prog)s --last-run           # Только последний запуск теста
  %(prog)s --run-id ABC123      # Конкретный run ID
  %(prog)s --list-runs          # Показать доступные run IDs
        """
    )
    parser.add_argument('--last-run', action='store_true',
                        help='Показать метрики только последнего запуска')
    parser.add_argument('--run-id', type=str,
                        help='Фильтровать по конкретному run ID')
    parser.add_argument('--list-runs', action='store_true',
                        help='Показать список доступных run IDs')
    
    args = parser.parse_args()
    
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}{'IMPALA Training Metrics Viewer':^60}{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")
    
    # Проверка соединения
    print(f"\n{Color.YELLOW}Проверка соединения с ClickHouse...{Color.END}")
    if not check_clickhouse_connection():
        print(f"{Color.RED}❌ SigNoz ClickHouse не запущен!{Color.END}")
        print(f"{Color.YELLOW}Запустите: cd examples/signoz-repo/deploy/docker && docker compose up -d{Color.END}")
        sys.exit(1)
    
    print(f"{Color.GREEN}✅ ClickHouse доступен{Color.END}")
    
    # Обработать --list-runs
    if args.list_runs:
        print(f"\n{Color.BOLD}{Color.CYAN}Доступные Run IDs:{Color.END}")
        runs = get_available_run_ids(limit=20)
        if runs:
            print(f"\n{'Run ID':<25} {'Start Time':<20} {'End Time':<20}")
            print("=" * 70)
            for run_id, start_time, end_time in runs:
                print(f"{run_id:<25} {start_time:<20} {end_time:<20}")
            print(f"\n{Color.YELLOW}💡 Используйте: python3 view_metrics.py --run-id <ID>{Color.END}")
        else:
            print(f"{Color.RED}❌ Не найдено run IDs с данными{Color.END}")
        sys.exit(0)
    
    # Определить run_id для фильтрации
    run_id = None
    if args.last_run:
        run_id = get_last_run_id()
        if run_id:
            print(f"\n{Color.GREEN}✅ Используется последний run: {Color.BOLD}{run_id}{Color.END}")
        else:
            print(f"\n{Color.RED}❌ Не найден последний run ID{Color.END}")
            sys.exit(1)
    elif args.run_id:
        run_id = args.run_id
        print(f"\n{Color.GREEN}✅ Фильтр по run ID: {Color.BOLD}{run_id}{Color.END}")
    else:
        print(f"\n{Color.YELLOW}ℹ️  Показываются все данные за последние 15 минут{Color.END}")
        print(f"{Color.YELLOW}   Используйте --last-run для просмотра только последнего запуска{Color.END}")
    
    # Получить список всех IMPALA метрик
    print(f"\n{Color.YELLOW}Поиск IMPALA метрик...{Color.END}")
    all_metrics = get_all_impala_metrics()
    
    if not all_metrics:
        print(f"\n{Color.RED}❌ IMPALA метрики не найдены!{Color.END}")
        print(f"{Color.YELLOW}Возможные причины:{Color.END}")
        print(f"  1. Тест ещё не запускался или не отправил данные")
        print(f"  2. OTEL Collector не работает")
        print(f"  3. Прошло более 15 минут с последнего запуска")
        print(f"\n{Color.YELLOW}Запустите тест:{Color.END}")
        print(f"  python3 test_impala_full_integration.py")
        sys.exit(1)
    
    print(f"{Color.GREEN}✅ Найдено {len(all_metrics)} IMPALA метрик{Color.END}")
    
    # =================================================================
    # СЕКЦИЯ 1: ОБУЧЕНИЕ (LEARNER) - САМОЕ ВАЖНОЕ
    # =================================================================
    print_section("🎓 ОБУЧЕНИЕ (Learner) - ГЛАВНЫЕ МЕТРИКИ")
    
    # Loss - самая важная метрика
    avg, min_val, max_val, samples = get_metric_stats('learner.loss', run_id=run_id)
    print_metric(
        "learner.loss (⭐ ГЛАВНАЯ МЕТРИКА)",
        avg, min_val, max_val, samples,
        good_threshold=10.0,  # loss < 10 = хорошо
        bad_threshold=20.0,   # loss > 20 = плохо
        lower_is_better=True
    )
    
    if samples > 0:
        print(f"{Color.YELLOW}💡 Совет: Loss должен постепенно УМЕНЬШАТЬСЯ. Если растёт - проблема!{Color.END}\n")
    
    # Agent NLL
    avg, min_val, max_val, samples = get_metric_stats('learner.agent_nll.mean', run_id=run_id)
    print_metric("learner.agent_nll.mean", avg, min_val, max_val, samples)
    
    # Prior NLL
    avg, min_val, max_val, samples = get_metric_stats('learner.prior_nll.mean', run_id=run_id)
    print_metric("learner.prior_nll.mean", avg, min_val, max_val, samples)
    
    # Importance Weights
    avg, min_val, max_val, samples = get_metric_stats('learner.importance_weights.mean', run_id=run_id)
    print_metric(
        "learner.importance_weights.mean",
        avg, min_val, max_val, samples,
        good_threshold=1.5,  # веса близкие к 1.0 = хорошо
        bad_threshold=3.0,   # веса > 3.0 = плохо
        lower_is_better=True
    )
    
    avg, min_val, max_val, samples = get_metric_stats('learner.importance_weights.max', run_id=run_id)
    print_metric(
        "learner.importance_weights.max",
        avg, min_val, max_val, samples,
        good_threshold=5.0,   # max веса < 5 = хорошо
        bad_threshold=10.0,   # max веса > 10 = плохо
        lower_is_better=True
    )
    
    # Update Duration
    avg, min_val, max_val, samples = get_metric_stats('learner.update.duration', run_id=run_id)
    print_metric(
        "learner.update.duration (секунды)",
        avg, min_val, max_val, samples,
        good_threshold=5.0,   # < 5 сек = быстро
        bad_threshold=10.0,   # > 10 сек = медленно
        lower_is_better=True
    )
    
    # Траектории
    trajectories_fetched = get_counter_value('learner.trajectories_fetched.total', run_id=run_id)
    print_counter("learner.trajectories_fetched.total", trajectories_fetched)
    
    # =================================================================
    # СЕКЦИЯ 2: ГЕНЕРАЦИЯ (ACTOR)
    # =================================================================
    print_section("🎬 ГЕНЕРАЦИЯ ТРАЕКТОРИЙ (Actor)")
    
    # Rewards
    avg, min_val, max_val, samples = get_metric_stats('actor.reward.mean')
    print_metric("actor.reward.mean", avg, min_val, max_val, samples)
    
    avg, min_val, max_val, samples = get_metric_stats('actor.reward.std')
    print_metric("actor.reward.std (разброс)", avg, min_val, max_val, samples)
    
    avg, min_val, max_val, samples = get_metric_stats('actor.reward.max')
    print_metric("actor.reward.max", avg, min_val, max_val, samples)
    
    # SMILES Validity - критично важная метрика
    avg, min_val, max_val, samples = get_metric_stats('actor.smiles.valid.ratio')
    print_metric(
        "actor.smiles.valid.ratio (⭐ КРИТИЧНО)",
        avg, min_val, max_val, samples,
        good_threshold=0.7,   # > 70% валидных = хорошо
        bad_threshold=0.5,    # < 50% валидных = плохо
        lower_is_better=False
    )
    
    if samples > 0:
        print(f"{Color.YELLOW}💡 Совет: Validity > 0.7 = OK, > 0.9 = отлично!{Color.END}\n")
    
    # Траектории
    trajectories_generated = get_counter_value('actor.trajectories_generated.total')
    print_counter("actor.trajectories_generated.total", trajectories_generated)
    
    # Generation Duration
    avg, min_val, max_val, samples = get_metric_stats('actor.trajectory_generation.duration')
    print_metric(
        "actor.trajectory_generation.duration (сек)",
        avg, min_val, max_val, samples,
        good_threshold=30.0,  # < 30 сек = быстро
        bad_threshold=60.0,   # > 60 сек = медленно
        lower_is_better=True
    )
    
    # =================================================================
    # СЕКЦИЯ 3: БАЛАНС И THROUGHPUT
    # =================================================================
    print_section("⚖️ БАЛАНС СИСТЕМЫ")
    
    print(f"{Color.BOLD}Траектории:{Color.END}")
    print(f"  Сгенерировано: {Color.GREEN}{trajectories_generated}{Color.END}")
    print(f"  Получено:      {Color.GREEN}{trajectories_fetched}{Color.END}")
    
    if trajectories_generated > 0 and trajectories_fetched > 0:
        ratio = trajectories_fetched / trajectories_generated
        if ratio > 0.8:
            status = f"{Color.GREEN}✅ ОТЛИЧНО - learner успевает обрабатывать{Color.END}"
        elif ratio > 0.5:
            status = f"{Color.YELLOW}⚠️ ПРИЕМЛЕМО - небольшое отставание{Color.END}"
        else:
            status = f"{Color.RED}❌ ПЛОХО - learner не успевает!{Color.END}"
        
        print(f"  Соотношение:   {ratio:.2%}  {status}")
    
    print()
    
    # =================================================================
    # ИТОГОВАЯ ОЦЕНКА
    # =================================================================
    print_section("🎯 ИТОГОВАЯ ОЦЕНКА")
    
    # Получить key метрики для общей оценки
    loss_avg, _, _, loss_samples = get_metric_stats('learner.loss')
    validity_avg, _, _, validity_samples = get_metric_stats('actor.smiles.valid.ratio')
    reward_avg, _, _, reward_samples = get_metric_stats('actor.reward.mean')
    
    issues = []
    
    if loss_samples == 0:
        issues.append(f"{Color.RED}❌ Нет данных по loss - learner не работает?{Color.END}")
    elif loss_avg > 20.0:
        issues.append(f"{Color.YELLOW}⚠️ Loss очень высокий ({loss_avg:.2f}) - модель плохо учится{Color.END}")
    
    if validity_samples == 0:
        issues.append(f"{Color.RED}❌ Нет данных по validity - actor не работает?{Color.END}")
    elif validity_avg < 0.5:
        issues.append(f"{Color.RED}❌ Validity слишком низкий ({validity_avg:.2%}) - модель генерирует мусор!{Color.END}")
    elif validity_avg < 0.7:
        issues.append(f"{Color.YELLOW}⚠️ Validity низковат ({validity_avg:.2%}) - желательно >0.7{Color.END}")
    
    if trajectories_generated == 0:
        issues.append(f"{Color.RED}❌ Не сгенерировано ни одной траектории - actors не запущены?{Color.END}")
    
    if trajectories_fetched == 0 and trajectories_generated > 0:
        issues.append(f"{Color.RED}❌ Траектории не доходят до learner - проблема с NATS?{Color.END}")
    
    if not issues:
        print(f"{Color.GREEN}{Color.BOLD}✅✅✅ ВСЁ ОТЛИЧНО - СИСТЕМА РАБОТАЕТ! ✅✅✅{Color.END}\n")
        print(f"{Color.GREEN}Модель обучается, траектории генерируются, всё в норме.{Color.END}\n")
    else:
        print(f"{Color.YELLOW}{Color.BOLD}⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ:{Color.END}\n")
        for issue in issues:
            print(f"  {issue}")
        print()
    
    # =================================================================
    # ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (если есть)
    # =================================================================
    other_metrics = [m for m in all_metrics if m not in [
        'learner.loss', 'learner.agent_nll.mean', 'learner.prior_nll.mean',
        'learner.importance_weights.mean', 'learner.importance_weights.max',
        'learner.update.duration', 'learner.trajectories_fetched.total',
        'actor.reward.mean', 'actor.reward.std', 'actor.reward.max',
        'actor.smiles.valid.ratio', 'actor.trajectories_generated.total',
        'actor.trajectory_generation.duration',
    ]]
    
    if other_metrics:
        print(f"\n{Color.BOLD}Дополнительные метрики ({len(other_metrics)}):{Color.END}")
        for metric in sorted(other_metrics)[:10]:  # Показать первые 10
            print(f"  • {metric}")
        
        if len(other_metrics) > 10:
            print(f"  ... и ещё {len(other_metrics) - 10} метрик")
    
    # =================================================================
    # FOOTER
    # =================================================================
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}")
    print(f"{Color.YELLOW}💡 Для более детального анализа откройте SigNoz:{Color.END}")
    print(f"   {Color.BLUE}http://localhost:8080{Color.END}")
    print(f"\n{Color.YELLOW}📖 Подробное руководство:{Color.END}")
    print(f"   {Color.CYAN}SIGNOZ_VIEWING_GUIDE.md{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}Прервано пользователем{Color.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Color.RED}Ошибка: {e}{Color.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

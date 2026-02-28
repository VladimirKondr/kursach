#!/usr/bin/env python3
"""
График loss в терминале с правильным расчётом из SigNoz ClickHouse.
Показывает реальную динамику обучения.
"""

import subprocess
import sys
from datetime import datetime

def query_clickhouse(query: str) -> list:
    """Выполнить запрос к ClickHouse."""
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
            print(f"Ошибка запроса: {result.stderr}")
            return []
    except Exception as e:
        print(f"Ошибка: {e}")
        return []


def get_loss_over_time(minutes: int = 30):
    """Получить loss по минутам с правильным расчётом (rate)."""
    query = f"""
    WITH loss_data AS (
        SELECT 
            intDiv(unix_milli, 60000) * 60 as time_bucket,
            metric_name,
            value
        FROM signoz_metrics.samples_v4
        WHERE metric_name IN ('learner.loss.sum', 'learner.loss.count')
        AND unix_milli >= (toUnixTimestamp(now()) - {minutes} * 60) * 1000
    ),
    aggregated AS (
        SELECT 
            time_bucket,
            max(sumIf(value, metric_name = 'learner.loss.sum')) as cumsum,
            max(sumIf(value, metric_name = 'learner.loss.count')) as cumcount
        FROM loss_data
        GROUP BY time_bucket
        ORDER BY time_bucket
    )
    SELECT 
        toDateTime(time_bucket) as time,
        round((cumsum - lagInFrame(cumsum, 1, 0) OVER (ORDER BY time_bucket)) / 
              nullIf(cumcount - lagInFrame(cumcount, 1, 0) OVER (ORDER BY time_bucket), 0), 2) as avg_loss,
        cumcount - lagInFrame(cumcount, 1, 0) OVER (ORDER BY time_bucket) as new_samples
    FROM aggregated
    WHERE new_samples > 0
    FORMAT TSVWithNames
    """
    
    result = query_clickhouse(query)
    return result


def plot_ascii(data_points):
    """Нарисовать простой ASCII график."""
    if not data_points:
        print("Нет данных для графика")
        return
    
    times = [d[0] for d in data_points]
    losses = [d[1] for d in data_points]
    
    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1
    
    height = 20
    width = min(len(losses), 60)
    
    print("\n" + "="*80)
    print("📉 ГРАФИК LOSS (правильный расчёт - rate, не cumulative)")
    print("="*80 + "\n")
    
    # Нормализуем данные для графика
    normalized = [(l - min_loss) / loss_range * height for l in losses]
    
    # Рисуем график сверху вниз
    for row in range(height, -1, -1):
        # Y-axis label
        y_value = min_loss + (row / height) * loss_range
        print(f"{y_value:8.1f} │", end="")
        
        # Plot line
        for i, norm_val in enumerate(normalized[:width]):
            if abs(norm_val - row) < 0.5:
                print("●", end="")
            elif row == 0:
                print("─", end="")
            else:
                print(" ", end="")
        print()
    
    # X-axis
    print("         └" + "─" * width)
    print(f"         {times[0]} → {times[-1]}")
    
    # Статистика
    print(f"\n📊 Статистика:")
    print(f"  Начальный loss: {losses[0]:.2f}")
    print(f"  Конечный loss:  {losses[-1]:.2f}")
    
    if losses[0] > 0:
        change_pct = ((losses[-1] - losses[0]) / losses[0]) * 100
        trend = "⬇️ УЛУЧШАЕТСЯ" if change_pct < 0 else "⬆️ УХУДШАЕТСЯ"
        print(f"  Изменение:      {change_pct:+.1f}% {trend}")
    
    print(f"  Мин loss:       {min_loss:.2f}")
    print(f"  Макс loss:      {max_loss:.2f}")
    print(f"  Точек данных:   {len(losses)}")
    
    # Тренд последних N точек
    if len(losses) >= 3:
        recent = losses[-3:]
        if recent[-1] < recent[0]:
            print(f"\n✅ Последние 3 точки: УЛУЧШЕНИЕ ({recent[0]:.1f} → {recent[-1]:.1f})")
        elif recent[-1] > recent[0]:
            print(f"\n⚠️  Последние 3 точки: УХУДШЕНИЕ ({recent[0]:.1f} → {recent[-1]:.1f})")
        else:
            print(f"\n➡️  Последние 3 точки: СТАБИЛЬНО (~{recent[-1]:.1f})")


def main():
    print("🔍 Получение данных из SigNoz ClickHouse...")
    
    result = get_loss_over_time(30)
    
    if not result or len(result) < 2:
        print("❌ Недостаточно данных для построения графика")
        print("   Запустите тест: python3 test_impala_full_integration.py")
        sys.exit(1)
    
    # Парсим результат (пропускаем заголовок)
    data_points = []
    for i, line in enumerate(result):
        if i == 0:  # Skip header
            continue
        parts = line.split('\t')
        if len(parts) >= 3:
            try:
                time_str = parts[0]
                loss = float(parts[1])
                samples = int(float(parts[2]))
                data_points.append((time_str, loss, samples))
            except (ValueError, IndexError):
                continue
    
    if not data_points:
        print("❌ Не удалось распарсить данные")
        sys.exit(1)
    
    # Показать табличку с данными
    print("\n📋 Данные loss по времени (правильный расчёт):\n")
    print(f"{'Время':<20} {'Loss':>10} {'Сэмплов':>10}")
    print("-" * 42)
    for time_str, loss, samples in data_points:
        print(f"{time_str:<20} {loss:>10.2f} {samples:>10}")
    
    # Построить график
    plot_ascii(data_points)
    
    print("\n" + "="*80)
    print("💡 Этот график показывает РЕАЛЬНЫЙ средний loss на каждом шаге,")
    print("   а не кумулятивную сумму, которую вы видели в SigNoz!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

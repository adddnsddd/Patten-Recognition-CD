"""Reproducible synthetic interactions and chronological data preparation."""
import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate(folder, seed=42, user_count=200, item_count=500, event_count=16000):
    """Labels come from latent preferences and noise, independently of GA fitness."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    categories = ['图书', '家居', '数码', '服饰', '运动']
    items = []
    for index in range(item_count):
        category = categories[index % len(categories)]
        items.append(dict(item_id=index + 1, name=f'{category}商品{index + 1:03d}',
                          category=category, tags=category,
                          price=round(float(rng.uniform(20, 800)), 2),
                          rating=round(float(rng.uniform(2.5, 5)), 1),
                          stock=int(rng.integers(0, 100)),
                          is_active=int(rng.random() > 0.03)))
    users = [dict(user_id=index + 1, price_min='', price_max='',
                  preferred_category='') for index in range(user_count)]
    # These hidden preferences are never passed to the recommender.
    tastes = rng.dirichlet(np.full(5, 0.35), size=user_count)
    typical_prices = rng.uniform(80, 650, size=user_count)
    item_appeal = rng.lognormal(0, 0.8, size=item_count)
    available = np.array([bool(item['is_active'] and item['stock'] > 0) for item in items])
    prices = np.array([item['price'] for item in items])
    category_indices = np.arange(item_count) % 5
    probabilities = []
    for index in range(user_count):
        price_affinity = np.exp(-((prices - typical_prices[index]) / 220) ** 2)
        weights = tastes[index, category_indices] * (0.2 + price_affinity) * item_appeal * available
        probabilities.append(weights / weights.sum())
    events = []
    start = datetime(2026, 1, 1)
    for index in range(event_count):
        user_index = index % user_count
        item_id = int(rng.choice(item_count, p=probabilities[user_index])) + 1
        action = str(rng.choice(['view', 'favorite', 'purchase'], p=[0.65, 0.22, 0.13]))
        when = start + timedelta(seconds=int(rng.integers(0, 60 * 86400)))
        events.append(dict(user_id=user_index + 1, item_id=item_id, action=action,
                           timestamp=when.isoformat(sep=' ')))
    events.sort(key=lambda event: event['timestamp'])
    write_csv(folder / 'users.csv', users, list(users[0]))
    write_csv(folder / 'items.csv', items, list(items[0]))
    write_csv(folder / 'behaviors.csv', events, list(events[0]))
    return dict(seed=seed, users=len(users), items=len(items), behaviors=len(events),
                categories=len(categories))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load(folder):
    folder = Path(folder)
    user_rows = read_csv(folder / 'users.csv')
    item_rows = read_csv(folder / 'items.csv')
    users = {int(row['user_id']): row for row in user_rows}
    if len(users) != len(user_rows):
        raise ValueError('用户编号重复')
    items = {}
    for row in item_rows:
        item = dict(row)
        item_id = int(row['item_id'])
        if item_id in items:
            raise ValueError('商品编号重复')
        item.update(item_id=item_id, price=float(row['price']), stock=int(row['stock']),
                    is_active=int(row['is_active']),
                    category=row['category'].strip() or '未知')
        rating = row['rating'].strip()
        item['rating'] = float(rating) if rating else None
        numeric = [item['price'], item['stock']]
        if (not all(np.isfinite(value) for value in numeric)
                or item['price'] < 0 or item['stock'] < 0
                or item['is_active'] not in (0, 1)
                or (item['rating'] is not None and
                    (not np.isfinite(item['rating']) or not 0 <= item['rating'] <= 5))):
            raise ValueError(f'商品{item_id}的价格、库存、状态或评分非法')
        items[item_id] = item
    raw = read_csv(folder / 'behaviors.csv')
    clean, seen, dropped = [], set(), Counter()
    for row in raw:
        try:
            uid, iid = int(row['user_id']), int(row['item_id'])
            action = row['action']
            # Naive CSV timestamps are explicitly interpreted as Asia/Shanghai.
            when = datetime.fromisoformat(row['timestamp'])
            if when.tzinfo is not None:
                raise ValueError('本地CSV要求统一使用北京时间且不带时区后缀')
            timestamp = when.isoformat(sep=' ')
            if uid not in users or iid not in items or action not in ('view', 'favorite', 'purchase'):
                raise ValueError('非法关联编号或动作')
        except (ValueError, KeyError, TypeError):
            dropped['invalid'] += 1
            continue
        event_time = when.replace(second=0, microsecond=0) if action == 'view' else when
        key = (uid, iid, action, event_time)
        if key in seen:
            dropped['duplicate'] += 1
            continue
        seen.add(key)
        clean.append(dict(user_id=uid, item_id=iid, action=action, timestamp=timestamp))
    clean.sort(key=lambda event: event['timestamp'])
    if len(clean) < 3:
        raise ValueError('至少需要三个不同时间段的行为记录')
    times = [event['timestamp'] for event in clean]
    # Move boundaries to the end of an equal-timestamp group.
    boundaries = []
    for fraction in (0.7, 0.85):
        index = max(1, int(len(clean) * fraction))
        while index < len(clean) and times[index] == times[index - 1]:
            index += 1
        boundaries.append(index)
    a, b = boundaries
    if not 0 < a < b < len(clean):
        raise ValueError('不同时间戳不足，无法划分训练、验证、测试区间')
    stats = dict(raw_records=len(raw), clean_records=len(clean), dropped=dict(dropped),
                 train_records=a, validation_records=b - a, test_records=len(clean) - b,
                 validation_start=times[a], test_start=times[b], timezone='Asia/Shanghai',
                 users=len(users), items=len(items),
                 known_categories=len({x['category'] for x in items.values()} - {'未知'}))
    return users, items, clean, clean[:a], clean[a:b], clean[b:], stats

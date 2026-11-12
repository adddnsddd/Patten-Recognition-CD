"""Train-only profiles, shared candidate pools and recommendation methods."""
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from .data import load
from .ga import Config, components, config_dict, optimize


class Engine:
    def __init__(self, folder):
        (self.users, self.items, self.events, self.train, self.validation,
         self.test, self.stats) = load(folder)
        self.ids = np.array(sorted(self.items), dtype=int)
        self.index = {int(iid): index for index, iid in enumerate(self.ids)}
        categories = sorted({item['category'] for item in self.items.values()})
        self.categories = categories
        self.category = np.array([self.items[int(iid)]['category'] for iid in self.ids])
        self.features = np.eye(len(categories))[[categories.index(cat) for cat in self.category]]
        self.similarity = self.features @ self.features.T
        self.price = np.array([self.items[int(iid)]['price'] for iid in self.ids])
        ratings = [item['rating'] for item in self.items.values() if item['rating'] is not None]
        fallback = float(np.median(ratings)) if ratings else 2.5
        self.quality = np.array([(self.items[int(iid)]['rating']
                                 if self.items[int(iid)]['rating'] is not None else fallback) / 5
                                for iid in self.ids])
        self.histories, self.past = defaultdict(list), defaultdict(list)
        interactions = defaultdict(set)
        for event in self.events:
            self.past[event['user_id']].append(event)
        for event in self.train:
            self.histories[event['user_id']].append(event)
            interactions[event['item_id']].add(event['user_id'])
        self.popularity = np.array([len(interactions[int(iid)]) for iid in self.ids])
        maximum = int(self.popularity.max()) if len(self.ids) else 0
        self.novelty = (1 - np.log1p(self.popularity) / np.log1p(maximum)
                        if maximum else np.ones(len(self.ids)))

    def profile(self, user_id):
        history = self.histories[user_id]
        vector = np.zeros(len(self.categories))
        purchases, other = [], []
        action_weight = dict(view=1, favorite=3, purchase=5)
        for event in history:
            index = self.index[event['item_id']]
            vector += action_weight[event['action']] * self.features[index]
            (purchases if event['action'] == 'purchase' else other).append(self.price[index])
        if not vector.any():
            preferred = self.users.get(user_id, {}).get('preferred_category', '')
            if preferred in self.categories:
                vector[self.categories.index(preferred)] = 1
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        price_history = purchases or other
        median_price = float(np.median(price_history if price_history else self.price))
        return vector, median_price, not bool(norm)

    def context(self, user_id, at=None, m=100, price_min=None, price_max=None, blocked=(),
                candidate_strategy='hybrid'):
        if not isinstance(m, (int, np.integer)) or m < 0:
            raise ValueError('候选数不能为负')
        if candidate_strategy not in ('hybrid', 'score'):
            raise ValueError('候选策略须为hybrid或score')
        at = at or self.stats['test_start']
        if at not in (self.stats['validation_start'], self.stats['test_start']):
            raise ValueError('本版本仅支持验证区间和测试区间起点，避免画像使用未来信息')
        user = self.users.get(user_id, {})
        low = price_min if price_min is not None else user.get('price_min')
        high = price_max if price_max is not None else user.get('price_max')
        low = 0.0 if low in ('', None) else float(low)
        high = float('inf') if high in ('', None) else float(high)
        if not np.isfinite(low) or np.isnan(high) or low < 0 or high < low:
            raise ValueError('单件商品价格范围非法')
        excluded = set(blocked)
        excluded.update(event['item_id'] for event in self.past[user_id]
                        if event['action'] == 'purchase' and event['timestamp'] < at)
        allowed = [index for index, iid in enumerate(self.ids)
                   if int(iid) not in excluded and self.items[int(iid)]['is_active']
                   and self.items[int(iid)]['stock'] > 0 and low <= self.price[index] <= high]
        vector, median, cold = self.profile(user_id)
        price_match = 1 / (1 + np.abs(self.price - median) / (median + 1e-6))
        scores = 0.6 * (self.features @ vector) + 0.25 * price_match + 0.15 * self.quality
        ranked = sorted(allowed, key=lambda index: (-scores[index], self.ids[index]))
        quota = m if candidate_strategy == 'score' else int(np.ceil(0.8 * m))
        candidates = ranked[:min(quota, len(ranked))]
        included = set(candidates)
        per_category = {cat: [index for index in ranked
                              if self.category[index] == cat and index not in included]
                        for cat in self.categories}
        while len(candidates) < min(m, len(allowed)):
            for cat in self.categories:
                if per_category[cat] and len(candidates) < min(m, len(allowed)):
                    index = per_category[cat].pop(0)
                    candidates.append(index)
        # Stable score / item_id ordering also guarantees tie-breaking inside GA.
        candidates.sort(key=lambda index: (-scores[index], self.ids[index]))
        return dict(user_id=user_id, at=at, allowed=allowed, candidates=candidates,
                    scores=scores, profile=vector, median_price=median, cold=cold,
                    price_range=[low, None if np.isinf(high) else high],
                    candidate_strategy=candidate_strategy)

    def recommend(self, user_id, method='ga', k=10, config=None, context=None, **kwargs):
        started = perf_counter()
        if (method not in ('popular', 'greedy', 'ga')
                or not isinstance(k, (int, np.integer)) or k < 0):
            raise ValueError('算法必须为popular、greedy或ga，列表长度不能为负')
        config = (config or Config()).validate()
        ctx = context if context is not None else self.context(user_id, **kwargs)
        if ctx['user_id'] != user_id:
            raise ValueError('候选池不属于当前用户')
        # Revalidate current stock/status, including changes since candidate generation.
        pool = ctx['allowed'] if ctx['cold'] and ctx['candidates'] else ctx['candidates']
        candidates = np.array([index for index in pool
                               if self.items[int(self.ids[index])]['stock'] > 0
                               and self.items[int(self.ids[index])]['is_active']], dtype=int)
        requested_k = k
        k = min(k, len(candidates))
        actual_method = 'popular' if ctx['cold'] else method
        scores = ctx['scores'][candidates]
        novelty = self.novelty[candidates]
        similarity = self.similarity[np.ix_(candidates, candidates)]
        history, optimization_seconds = [], 0.0
        if actual_method == 'ga' and k:
            order, history, optimization_seconds = optimize(scores, novelty, similarity, k, config)
        elif actual_method == 'popular':
            order = sorted(range(len(candidates)), key=lambda index:
                           (-self.popularity[candidates[index]], self.ids[candidates[index]]))[:k]
        else:
            order = np.argsort(-scores, kind='stable')[:k].tolist()
        r, d, v = components(order, scores, novelty, similarity)[0]
        output = []
        for rank, local in enumerate(order, 1):
            index = candidates[local]
            item = dict(self.items[int(self.ids[index])])
            item.update(rank=rank, relevance=float(scores[local]),
                        novelty=float(novelty[local]),
                        reason=('训练期热门商品（冷启动回退）' if ctx['cold'] else
                                '匹配偏好类别' if ctx['profile'] @ self.features[index] > 0.3
                                else '提供类别补充'))
            output.append(item)
        return dict(user_id=user_id, requested_method=method, method=actual_method,
                    cold_start=ctx['cold'], status='ok' if output else '暂无符合条件的物品',
                    requested_at=ctx['at'], returned=len(output), requested_length=requested_k,
                    length_note=('可用商品不足，已返回全部可用商品' if len(output) < requested_k else ''),
                    candidate_strategy=ctx.get('candidate_strategy', 'hybrid'),
                    candidate_count=len(candidates),
                    candidate_categories=dict(Counter(str(self.category[i]) for i in candidates)),
                    profile=dict(zip(self.categories, ctx['profile'].tolist())),
                    median_price=ctx['median_price'], price_range=ctx['price_range'],
                    R=float(r), D=float(d), V=float(v),
                    F=float(np.array([r, d, v]) @ config.weights), items=output,
                    optimization_seconds=optimization_seconds,
                    request_seconds=perf_counter() - started,
                    history=history, config=config_dict(config))

    def truth(self, user_id, interval, context):
        allowed = {int(self.ids[index]) for index in context['allowed']}
        return {event['item_id'] for event in interval
                if event['user_id'] == user_id and event['action'] in ('favorite', 'purchase')
                and event['item_id'] in allowed}

"""Ordered subset search with elitism and prefix crossover."""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np


@dataclass
class Config:
    population: int = 60
    generations: int = 100
    crossover: float = 0.8
    mutation: float = 0.2
    patience: int = 15
    tolerance: float = 0.0001
    weights: tuple = (0.7, 0.2, 0.1)
    seed: int = 0
    replacement_mutation: bool = True

    def validate(self):
        if (any(not isinstance(value, (int, np.integer)) for value in
                (self.population, self.generations, self.patience, self.seed))
                or self.population < 3 or self.generations < 1 or self.patience < 1):
            raise ValueError('种群至少为3，代数和停滞代数必须为正')
        if not (0 <= self.crossover <= 1 and 0 <= self.mutation <= 1):
            raise ValueError('交叉和变异概率必须位于0到1')
        if not np.isfinite(self.tolerance) or self.tolerance < 0 or self.seed < 0:
            raise ValueError('随机种子及停止阈值必须非负且有限')
        weights = np.asarray(self.weights, dtype=float)
        if (weights.shape != (3,) or not np.all(np.isfinite(weights))
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
            raise ValueError('三个目标权重须非负且和为1')
        return self


def components(lists, scores, novelty, similarity):
    lists = np.asarray(lists, dtype=int)
    if lists.ndim == 1:
        lists = lists[None, :]
    count, k = lists.shape
    if not k:
        return np.zeros((count, 3))
    discount = 1 / np.log2(np.arange(k) + 2)
    relevance = (scores[lists] @ discount) / discount.sum()
    fresh = (novelty[lists] @ discount) / discount.sum()
    diversity = np.zeros(count)
    if k > 1:
        a, b = np.triu_indices(k, 1)
        diversity = np.mean(1 - similarity[lists[:, a], lists[:, b]], axis=1)
    return np.column_stack((relevance, diversity, fresh))


def repair(values, count, k, rng):
    k = min(k, count)
    if k <= 0:
        return []
    result, used = [], set()
    for value in values:
        value = int(value)
        if 0 <= value < count and value not in used:
            result.append(value)
            used.add(value)
        if len(result) == k:
            return result
    remaining = [value for value in range(count) if value not in used]
    if len(result) < k:
        result.extend(int(value) for value in rng.choice(remaining, k - len(result), replace=False))
    return result


def optimize(scores, novelty, similarity, k, config):
    config.validate()
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    count = len(scores)
    k = min(k, count)
    if not k:
        return [], [], 0.0
    greedy = np.argsort(-scores, kind='stable')[:k]
    population = np.array([greedy] + [rng.choice(count, k, replace=False)
                                     for _ in range(config.population - 1)])
    weights = np.array(config.weights)
    history, stall, previous_best = [], 0, None
    best, best_fitness = greedy.copy(), -np.inf
    for generation in range(config.generations + 1):
        fitness = components(population, scores, novelty, similarity) @ weights
        winner = int(np.argmax(fitness))
        if fitness[winner] > best_fitness:
            best, best_fitness = population[winner].copy(), float(fitness[winner])
        history.append(dict(generation=generation, best=best_fitness,
                            mean=float(fitness.mean())))
        if previous_best is not None:
            stall = stall + 1 if best_fitness - previous_best <= config.tolerance else 0
        previous_best = best_fitness
        if generation == config.generations or stall >= config.patience:
            break
        elites = np.argsort(-fitness, kind='stable')[:2]
        next_generation = [population[index].copy().tolist() for index in elites]

        def select():
            contenders = rng.choice(len(population), 3, replace=False)
            return population[contenders[np.argmax(fitness[contenders])]].copy().tolist()

        while len(next_generation) < config.population:
            first, second = select(), select()
            children = [first.copy(), second.copy()]
            if k > 1 and rng.random() < config.crossover:
                cut = int(rng.integers(1, k))
                children = [repair(first[:cut] + second, count, k, rng),
                            repair(second[:cut] + first, count, k, rng)]
            for child in children:
                if rng.random() < config.mutation:
                    remaining = np.setdiff1d(np.arange(count), child)
                    if k > 1 and (not config.replacement_mutation or not len(remaining)
                                  or rng.random() < 0.5):
                        a, b = rng.choice(k, 2, replace=False)
                        child[a], child[b] = child[b], child[a]
                    elif len(remaining) and config.replacement_mutation:
                        child[int(rng.integers(k))] = int(rng.choice(remaining))
                next_generation.append(repair(child, count, k, rng))
                if len(next_generation) == config.population:
                    break
        population = np.array(next_generation, dtype=int)
    return best.tolist(), history, perf_counter() - started


def config_dict(config):
    return asdict(config)

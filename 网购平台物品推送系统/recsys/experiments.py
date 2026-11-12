"""Six predeclared, checkpointed experiments with validation-only selection.

Run: python -m recsys.experiments --data data --output outputs/suite
Each finished variant is saved immediately, so an interrupted run can resume.
"""
import argparse
import copy
import hashlib
import json
import platform
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .core import Engine
from .data import read_csv, write_csv
from .ga import Config, config_dict


METRICS = ('precision', 'recall', 'ndcg', 'R', 'D', 'V', 'F', 'coverage',
           'candidate_recall', 'optimization_seconds', 'request_seconds',
           'generations', 'returned')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def ranking_metrics(item_ids, truth, k=10):
    hits = len(set(item_ids[:k]) & truth)
    discount = 1 / np.log2(np.arange(k) + 2)
    dcg = sum(float(discount[index]) for index, iid in enumerate(item_ids[:k]) if iid in truth)
    ideal = float(discount[:min(k, len(truth))].sum())
    return dict(precision=hits / k, recall=hits / len(truth) if truth else None,
                ndcg=dcg / ideal if ideal else None)


def aggregate(seed_runs):
    """Macro-average users first; sample standard deviation is across GA seeds."""
    return {name: dict(mean=float(np.mean([run[name] for run in seed_runs])),
                       std=float(np.std([run[name] for run in seed_runs], ddof=1))
                       if len(seed_runs) > 1 else 0.0)
            for name in METRICS}


def select_variant(variants):
    # Time is deliberately not used to break ties, since timing is nondeterministic.
    if any(variant['split'] != 'validation' for variant in variants):
        raise ValueError('只能根据验证集选择配置')
    return min(variants, key=lambda variant: (-variant['summary']['ndcg']['mean'],
                                             -variant['summary']['F']['mean'], variant['name']))


def environment():
    info = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                processor=platform.processor(), logical_cpu_count=None,
                physical_memory_bytes=None)
    import os
    info['logical_cpu_count'] = os.cpu_count()
    if sys.platform == 'win32':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual',
                 'avail_virtual', 'avail_extended')]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            info['physical_memory_bytes'] = memory.total_phys
    return info


def cohort(engine, max_users):
    eligible, contexts, truths = [], {}, {}
    for uid in sorted(engine.users):
        if len(engine.histories[uid]) < 10:
            continue
        val = engine.context(uid, at=engine.stats['validation_start'])
        test = engine.context(uid, at=engine.stats['test_start'])
        val_truth = engine.truth(uid, engine.validation, val)
        test_truth = engine.truth(uid, engine.test, test)
        if len(val['candidates']) >= 10 and len(test['candidates']) >= 10 and val_truth and test_truth:
            eligible.append(uid)
            contexts[uid] = val
            truths[uid] = val_truth
    selected = eligible[:max_users]
    if not selected:
        raise ValueError('没有符合主实验条件的用户')
    case_users, dominant_categories = [], set()
    for uid in selected:
        dominant = int(np.argmax(contexts[uid]['profile']))
        if dominant not in dominant_categories:
            case_users.append(uid)
            dominant_categories.add(dominant)
        if len(case_users) >= 3:
            break
    for uid in selected:
        if len(case_users) >= min(3, len(selected)):
            break
        if uid not in case_users:
            case_users.append(uid)
    return dict(users_before_filter=len(engine.users), eligible_count=len(eligible),
                selected_users=selected, case_users=case_users,
                selection='按用户编号升序取前max_users名；案例优先选择画像主类别不同的用户',
                eligibility='训练行为>=10，验证和测试均有完整有效正反馈，两请求候选>=10；'
                            '测试标签仅用于预先约定的用户资格筛选，不参与验证选参')


def _row(result, truth, ctx, engine, seed, elapsed):
    item_ids = [item['item_id'] for item in result['items']]
    candidate_ids = {int(engine.ids[index]) for index in ctx['candidates']}
    return dict(user_id=result['user_id'], seed=seed, item_ids=item_ids,
                **ranking_metrics(item_ids, truth),
                **{name: result[name] for name in ('R', 'D', 'V', 'F')},
                candidate_recall=len(candidate_ids & truth) / len(truth),
                optimization_seconds=result['optimization_seconds'],
                request_seconds=elapsed, engine_request_seconds=result['request_seconds'],
                generations=max(0, len(result['history']) - 1),
                returned=result['returned'], candidate_count=result['candidate_count'],
                candidate_categories=result['candidate_categories'],
                positive_count=len(truth), invalid_count=0)


def assert_legal(result, ctx, engine):
    ids = [item['item_id'] for item in result['items']]
    allowed = {int(engine.ids[index]) for index in ctx['allowed']}
    candidates = {int(engine.ids[index]) for index in ctx['candidates']}
    if len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise AssertionError('出现重复物品或违反硬约束')
    if not result['cold_start'] and not set(ids) <= candidates:
        raise AssertionError('推荐物品不在候选池')
    if any(not item['is_active'] or item['stock'] <= 0 for item in result['items']):
        raise AssertionError('推荐包含下架或无库存物品')
    best = [entry['best'] for entry in result['history']]
    if any(b + 1e-12 < a for a, b in zip(best, best[1:])):
        raise AssertionError('历史最优适应度下降')


def run_variant(engine, users, seeds, config, split, method, m, strategy, case_users):
    at = engine.stats['validation_start' if split == 'validation' else 'test_start']
    interval = engine.validation if split == 'validation' else engine.test
    contexts = {uid: engine.context(uid, at=at, m=m, candidate_strategy=strategy) for uid in users}
    truths = {uid: engine.truth(uid, interval, contexts[uid]) for uid in users}
    if any(not truths[uid] for uid in users):
        raise ValueError('主实验用户存在空真值，不能混入宏平均')
    catalog = {int(engine.ids[index]) for ctx in contexts.values() for index in ctx['allowed']}
    # Warm up once outside all timed observations, including the baseline variants.
    engine.recommend(users[0], method=method, context=contexts[users[0]], config=config)
    greedy_scores = {uid: engine.recommend(uid, method='greedy', context=contexts[uid],
                                         config=config)['F'] for uid in users}
    rows, seed_runs, cases = [], [], {}
    run_seeds = seeds if method == 'ga' else [None]
    for seed in run_seeds:
        run_rows, covered = [], set()
        for uid in users:
            start = perf_counter()
            # Rebuild with identical rules; measure candidate generation + recommendation.
            ctx = engine.context(uid, at=at, m=m, candidate_strategy=strategy)
            if ctx['candidates'] != contexts[uid]['candidates']:
                raise AssertionError('共享候选池不稳定')
            result = engine.recommend(uid, method=method, context=ctx,
                                      config=replace(config, seed=seed if seed is not None else 0))
            elapsed = perf_counter() - start
            assert_legal(result, ctx, engine)
            if method == 'ga' and result['F'] + 1e-12 < greedy_scores[uid]:
                raise AssertionError('遗传算法最终F低于初始贪心F')
            row = _row(result, truths[uid], ctx, engine, seed, elapsed)
            rows.append(row)
            run_rows.append(row)
            covered.update(row['item_ids'])
            if seed == run_seeds[0] and uid in case_users:
                cases[str(uid)] = result
        run = dict(seed=seed, coverage=len(covered) / len(catalog))
        for name in METRICS:
            if name != 'coverage':
                run[name] = float(np.mean([row[name] for row in run_rows]))
        seed_runs.append(run)
    optimization_times = [row['optimization_seconds'] for row in rows]
    request_times = [row['request_seconds'] for row in rows]
    return dict(split=split, method=method, config=config_dict(config), m=m,
                candidate_strategy=strategy, users=users, seeds=run_seeds,
                summary=aggregate(seed_runs), seed_runs=seed_runs, per_user=rows,
                optimization_p50_seconds=float(np.percentile(optimization_times, 50)),
                optimization_p95_seconds=float(np.percentile(optimization_times, 95)),
                optimization_p99_seconds=float(np.percentile(optimization_times, 99)),
                request_p95_seconds=float(np.percentile(request_times, 95)),
                convergence=next(iter(cases.values()))['history'] if cases else [],
                case_results=cases, invalid_count=0,
                timing='预热一次；request包含画像/候选构造及推荐，不包含引擎加载、绘图、文件写入；'
                       'optimization仅为遗传优化。所有方法使用相同候选规则，'
                       '画像和商品相似度在引擎中预先建立。')


def _edge_result(name, engine, uid, config, **kwargs):
    result = engine.recommend(uid, config=config, **kwargs)
    item_ids = [item['item_id'] for item in result['items']]
    passed = (len(item_ids) == len(set(item_ids)) and
              all(item['stock'] > 0 and item['is_active'] for item in result['items']))
    return dict(name=name, passed=passed, returned=result['returned'],
                cold_start=result['cold_start'], method=result['method'],
                status=result['status'], candidate_count=result['candidate_count'],
                item_ids=item_ids, R=result['R'], D=result['D'], V=result['V'], F=result['F'],
                details=dict(profile=result['profile'], history_length=len(result['history'])),
                notes='边界场景不计入主实验的排序指标均值')


def edge_scenarios(engine, user_id, seed):
    config = Config(seed=seed)
    records = []
    fresh = copy.deepcopy(engine)
    cold_uid = max(engine.users) + 1000
    result = _edge_result('cold_no_history', fresh, cold_uid, config)
    result['passed'] &= result['cold_start'] and result['method'] == 'popular'
    records.append(result)
    for count in (1, 2, 3):
        sparse = copy.deepcopy(engine)
        sparse.histories[user_id] = sparse.histories[user_id][:count]
        sparse.past[user_id] = list(sparse.histories[user_id])
        result = _edge_result(f'sparse_{count}_events', sparse, user_id, config)
        result['details']['training_events'] = count
        result['notes'] += '；该用户建模及已购过滤历史均限制为前n条，其他用户及总体热度固定'
        records.append(result)
    for count in (0, 5, 10):
        result = _edge_result(f'candidate_{count}', engine, user_id, config, m=count)
        result['passed'] &= result['returned'] == count
        records.append(result)
    # Test actual all-missing rating imputation via a fresh Engine, not array surgery.
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        write_csv(folder / 'users.csv', engine.users.values(), list(next(iter(engine.users.values()))))
        items = [dict(item, rating='') for item in engine.items.values()]
        write_csv(folder / 'items.csv', items, list(items[0]))
        write_csv(folder / 'behaviors.csv', engine.events, list(engine.events[0]))
        missing = Engine(folder)
        result = _edge_result('all_ratings_missing', missing, user_id, config)
        result['passed'] &= bool(np.allclose(missing.quality, 0.5))
        result['details']['imputed_quality'] = float(missing.quality[0])
        records.append(result)
    changed = copy.deepcopy(engine)
    ctx = changed.context(user_id)
    chosen = changed.recommend(user_id, context=ctx, config=config)['items'][0]['item_id']
    changed.items[chosen]['stock'] = 0
    result = _edge_result('stock_changes_after_candidates', changed, user_id, config, context=ctx)
    result['passed'] &= chosen not in result['item_ids']
    result['details']['newly_unavailable_item_id'] = chosen
    records.append(result)
    if not all(record['passed'] for record in records):
        raise AssertionError('边界场景验证失败')
    return records


def _config_plan():
    default = Config()
    parameters = [('default', default)]
    for value in (30, 100):
        parameters.append((f'population_{value}', replace(default, population=value)))
    for value in (0.6, 0.9):
        parameters.append((f'crossover_{value}', replace(default, crossover=value)))
    for value in (0.05, 0.4):
        parameters.append((f'mutation_{value}', replace(default, mutation=value)))
    weights = [('weight_1_0_0', replace(default, weights=(1.0, 0.0, 0.0))),
               ('default', default),
               ('weight_05_03_02', replace(default, weights=(0.5, 0.3, 0.2)))]
    return default, parameters, weights


def run_suite(engine, output_dir, max_users=20, seeds=None):
    seeds = list(range(10)) if seeds is None else list(seeds)
    if max_users < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('评价用户数须为正；种子列表须非空且不重复')
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError('随机种子必须是非负整数')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    default, parameter_plan, weight_plan = _config_plan()
    selected = cohort(engine, max_users)
    users = selected['selected_users']
    source_files = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in ('core.py', 'ga.py', 'data.py', 'experiments.py')}
    identity = dict(protocol_version=1, cohort=selected, seeds=seeds,
                    data_hash=fingerprint([engine.users, engine.items, engine.events]),
                    implementation_hashes=source_files, default=config_dict(default),
                    parameters=[dict(name=name, config=config_dict(cfg)) for name, cfg in parameter_plan],
                    weights=[dict(name=name, config=config_dict(cfg)) for name, cfg in weight_plan],
                    ablation=['swap_only', 'score_candidates'], scales=[50, 100, 200],
                    selection_rule='验证NDCG降序，F降序，variant名称升序；不使用计时打破同分',
                    test_policy='原默认参数测试结果已在前一轮查看；本轮只依据验证实验选择配置，'
                                '锁定后执行预定测试比较，不宣称本测试集从未被查看')
    identity_hash = fingerprint(identity)
    manifest_file = output / 'manifest.json'
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text(encoding='utf-8'))
        if old['identity_hash'] != identity_hash:
            raise ValueError('实验目录已有不同数据/代码/配置的冻结计划，请使用新的--output目录')
        manifest = old
    else:
        manifest = dict(identity_hash=identity_hash, plan=identity,
                        frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                        environment=environment(), data_split=engine.stats,
                        static_metadata='所有商品属性在模拟开始即已知，主实验中保持静态',
                        metric_policy='Precision固定分母10；Recall和IDCG使用完整有效正反馈集合，'
                                      '不按候选池截断；先用户宏平均再种子均值/样本标准差')
        save_json(manifest_file, manifest)
    write_csv(output / 'behaviors_clean.csv', engine.events, list(engine.events[0]))
    save_json(output / 'cohort.json', selected)
    save_json(output / 'data_audit.json', engine.stats)
    memory_cache = {}

    def variant(name, cfg=default, split='validation', method='ga', m=100, strategy='hybrid'):
        key = fingerprint(dict(config=config_dict(cfg), split=split, method=method,
                               m=m, strategy=strategy, seeds=seeds, identity_hash=identity_hash))[:20]
        path = output / 'variants' / f'{key}.json'
        if key in memory_cache:
            result = copy.deepcopy(memory_cache[key])
        elif path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            memory_cache[key] = result
        else:
            print(f'开始 {name}: {split}, {method}, M={m}', flush=True)
            result = run_variant(engine, users, seeds, cfg, split, method, m, strategy,
                                 selected['case_users'])
            save_json(path, result)
            memory_cache[key] = result
        result = copy.deepcopy(result)
        # Large per-user raw records live in the checkpoint, with a verified relative locator.
        result['per_user_file'] = path.relative_to(output).as_posix()
        result['raw_record_count'] = len(result['per_user'])
        result.pop('per_user')
        result['name'] = name
        print(f'完成 {name}: NDCG={result["summary"]["ndcg"]["mean"]:.6f} '
              f'D={result["summary"]["D"]["mean"]:.4f}', flush=True)
        return result

    experiments = dict(parameters=[], weights=[], ablation=[], scale=[], edge=[], baseline=[])
    partial = dict(meta=manifest, cohort=selected, locked_config=None,
                   experiments=experiments, cases=[])

    def checkpoint():
        save_json(output / 'suite.partial.json', partial)

    for name, cfg in parameter_plan:
        experiments['parameters'].append(variant(name, cfg))
        checkpoint()
    for name, cfg in weight_plan:
        experiments['weights'].append(variant(name, cfg))
        checkpoint()
    choices = {entry['name']: entry for entry in experiments['parameters'] + experiments['weights']}
    winner = select_variant(list(choices.values()))
    config_values = dict(winner['config'])
    config_values['weights'] = tuple(config_values['weights'])
    locked = Config(**config_values)
    lock = dict(source_split='validation', variant_name=winner['name'],
                rule=identity['selection_rule'], config=config_dict(locked),
                validation_summary=winner['summary'], identity_hash=identity_hash,
                locked_before_final_test=True)
    lock_file = output / 'locked_config.json'
    if lock_file.exists():
        previous = json.loads(lock_file.read_text(encoding='utf-8'))
        if fingerprint(previous) != fingerprint(lock):
            raise ValueError('已有锁定配置不同；禁止在同一实验计划下改写测试配置')
    else:
        save_json(lock_file, lock)
    partial['locked_config'] = lock
    experiments['edge'] = edge_scenarios(engine, users[0], seeds[0])
    checkpoint()
    experiments['ablation'].append(variant('default', default))
    experiments['ablation'].append(variant('swap_only', replace(default, replacement_mutation=False)))
    experiments['ablation'].append(variant('score_candidates', default, strategy='score'))
    checkpoint()
    for m in (50, 100, 200):
        record = variant(f'candidates_{m}', default, m=m)
        # Same seed, same context, same input must yield identical list and convergence.
        uid = users[0]
        first = engine.recommend(uid, at=engine.stats['validation_start'], m=m,
                                 config=replace(default, seed=seeds[0]))
        second = engine.recommend(uid, at=engine.stats['validation_start'], m=m,
                                  config=replace(default, seed=seeds[0]))
        record['reproducible'] = (first['history'] == second['history'] and
                                  [i['item_id'] for i in first['items']] ==
                                  [i['item_id'] for i in second['items']])
        if not record['reproducible']:
            raise AssertionError('同一随机种子结果不一致')
        experiments['scale'].append(record)
        checkpoint()
    # Final test comes only after selection and the immutable lock have been saved.
    for method in ('popular', 'greedy', 'ga'):
        experiments['baseline'].append(variant(method if method != 'ga' else 'ga_default',
                                               default, split='test', method=method))
        checkpoint()
    experiments['baseline'].append(variant('ga_locked', locked, split='test'))
    locked_test = experiments['baseline'][-1]
    baseline_by_method = {entry['method']: entry for entry in experiments['baseline'][:3]}
    for uid in selected['case_users']:
        ctx = engine.context(uid)
        results = {method: baseline_by_method[method]['case_results'][str(uid)]
                   for method in ('popular', 'greedy', 'ga')}
        results['ga_locked'] = locked_test['case_results'][str(uid)]
        partial['cases'].append(dict(user_id=uid, profile=results['ga']['profile'],
                                     candidate_count=len(ctx['candidates']),
                                     candidate_categories=results['ga']['candidate_categories'],
                                     results=results))
    partial['meta'] = dict(manifest, elapsed_seconds=perf_counter() - started,
                           completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                           complete=True, aggregate_note='确定性基线只执行一次，std=0；'
                           'GA标准差为每种子用户宏平均的样本标准差。')
    flat = []
    for group, records in experiments.items():
        if group == 'edge':
            continue
        for record in records:
            row = dict(group=group, name=record['name'], split=record['split'],
                       method=record['method'], m=record['m'],
                       candidate_strategy=record['candidate_strategy'],
                       population=record['config']['population'],
                       crossover=record['config']['crossover'],
                       mutation=record['config']['mutation'],
                       weights=json.dumps(record['config']['weights']),
                       replacement_mutation=record['config']['replacement_mutation'],
                       optimization_p95_seconds=record['optimization_p95_seconds'],
                       request_p95_seconds=record['request_p95_seconds'])
            for name, summary in record['summary'].items():
                row[name + '_mean'] = summary['mean']
                row[name + '_std'] = summary['std']
            flat.append(row)
    write_csv(output / 'summary.csv', flat, list(flat[0]))
    save_json(output / 'suite.json', partial)
    print(f'全部实验完成，结果目录：{output}', flush=True)
    return partial


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='预设六组实验、验证集选参和锁定测试')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs/suite'))
    parser.add_argument('--max-users', type=int, default=20)
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = parser.parse_args()
    try:
        run_suite(Engine(args.data), args.output, args.max_users,
                  [int(value) for value in args.seeds.split(',')])
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'错误：{error}\n')


if __name__ == '__main__':
    main()

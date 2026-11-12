"""CLI entry point. Run from this project folder."""
import argparse
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from recsys.core import Engine
from recsys.data import generate
from recsys.ga import Config, config_dict


ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def ranking_metrics(result, truth):
    ids = [item['item_id'] for item in result['items']]
    hits = len(set(ids) & truth)
    discount = 1 / np.log2(np.arange(10) + 2)
    dcg = sum(discount[index] for index, iid in enumerate(ids[:10]) if iid in truth)
    ideal = float(discount[:min(10, len(truth))].sum())
    return dict(precision=hits / 10, recall=hits / len(truth) if truth else None,
                ndcg=dcg / ideal if ideal else None)


def evaluate(engine, seeds, max_users, output):
    if not seeds or len(set(seeds)) != len(seeds) or max_users < 1:
        raise ValueError('种子列表须非空且不重复，评价用户数须为正')
    train_counts = {uid: len(events) for uid, events in engine.histories.items()}
    contexts, truths = {}, {}
    eligible = []
    for uid in sorted(engine.users):
        if train_counts.get(uid, 0) < 10:
            continue
        val = engine.context(uid, at=engine.stats['validation_start'])
        test = engine.context(uid)
        val_truth = engine.truth(uid, engine.validation, val)
        test_truth = engine.truth(uid, engine.test, test)
        if (len(val['candidates']) >= 10 and len(test['candidates']) >= 10
                and val_truth and test_truth):
            eligible.append(uid)
            contexts[uid], truths[uid] = test, test_truth
    users = eligible[:max_users]
    if not users:
        raise ValueError('没有符合主实验条件的用户，请检查数据和时间划分')
    contexts = {uid: contexts[uid] for uid in users}
    config = Config()
    # Warm up once; exclude all loading, plotting and JSON I/O from timing.
    engine.recommend(users[0], context=contexts[users[0]], config=config)
    rows, runs = [], []
    catalog = {int(engine.ids[index]) for ctx in contexts.values() for index in ctx['allowed']}
    candidate_recall = float(np.mean([
        len({int(engine.ids[i]) for i in contexts[uid]['candidates']} & truths[uid]) / len(truths[uid])
        for uid in users]))
    for method in ('popular', 'greedy', 'ga'):
        for seed in (seeds if method == 'ga' else [None]):
            run_rows, covered = [], set()
            for uid in users:
                result = engine.recommend(uid, method=method, context=contexts[uid],
                                          config=replace(config, seed=seed or 0))
                metrics = ranking_metrics(result, truths[uid])
                row = dict(user_id=uid, method=method, seed=seed, **metrics,
                           R=result['R'], D=result['D'], V=result['V'], F=result['F'],
                           optimization_seconds=result['optimization_seconds'],
                           request_seconds=result['request_seconds'],
                           generations=len(result['history']) - 1 if result['history'] else 0,
                           returned=result['returned'],
                           candidate_count=result['candidate_count'],
                           positive_count=len(truths[uid]),
                           item_ids=[item['item_id'] for item in result['items']])
                rows.append(row)
                run_rows.append(row)
                covered.update(row['item_ids'])
            names = ('precision', 'recall', 'ndcg', 'R', 'D', 'V', 'F',
                     'optimization_seconds', 'request_seconds')
            runs.append(dict(method=method, seed=seed,
                             **{name: float(np.mean([row[name] for row in run_rows])) for name in names},
                             coverage=len(covered) / len(catalog)))
            print(f'{method:7s} seed={str(seed):>4s} NDCG={runs[-1]["ndcg"]:.4f} '
                  f'D={runs[-1]["D"]:.4f}', flush=True)
    summary = {}
    for method in ('popular', 'greedy', 'ga'):
        subset = [run for run in runs if run['method'] == method]
        summary[method] = {name: dict(mean=float(np.mean([run[name] for run in subset])),
                                    std=float(np.std([run[name] for run in subset], ddof=1))
                                    if len(subset) > 1 else 0.0)
                           for name in subset[0] if name not in ('method', 'seed')}
    ga_times = [row['optimization_seconds'] for row in rows if row['method'] == 'ga']
    report = dict(note='默认参数测试集基线比较；尚未进行验证集调参和完整六组实验',
                  split=engine.stats, users_before_filter=len(engine.users),
                  eligible_count=len(eligible), selected_users=users,
                  selection='按user_id升序选前max_users名合格用户',
                  seeds=seeds, config=config_dict(config), candidate_recall=candidate_recall,
                  optimization_p95_seconds=float(np.percentile(ga_times, 95)),
                  summary=summary, seed_runs=runs, per_user=rows,
                  environment=dict(python=sys.version, numpy=np.__version__,
                                   platform=platform.platform(), processor=platform.processor()),
                  timing_note='优化耗时只包含GA；完整request耗时在共享候选池已缓存条件下测量，'
                              '不包含画像及候选生成；端到端耗时请参考单用户recommend输出。')
    save_json(output, report)
    save_json(ROOT / 'outputs' / 'data_audit.json', engine.stats)
    print(f'结果保存至 {output}，用户数={len(users)}，候选召回率={candidate_recall:.4f}')


def main():
    # Keep redirected logs and Windows terminals consistent with saved UTF-8 files.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='基于遗传算法的网购推荐离线仿真')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('generate', help='生成教学数据（将覆盖data下三张CSV）')
    make.add_argument('--seed', type=int, default=42)
    recommend = sub.add_parser('recommend', help='生成一名用户的推荐结果')
    recommend.add_argument('--user', type=int, default=1)
    recommend.add_argument('--method', choices=['popular', 'greedy', 'ga'], default='ga')
    recommend.add_argument('--seed', type=int, default=0)
    recommend.add_argument('--price-min', type=float)
    recommend.add_argument('--price-max', type=float)
    recommend.add_argument('--k', type=int, default=10)
    recommend.add_argument('--candidates', type=int, default=100)
    recommend.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'demo.json')
    experiment = sub.add_parser('evaluate', help='测试集三种算法对照；固定默认参数')
    experiment.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    experiment.add_argument('--max-users', type=int, default=20)
    experiment.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'evaluation.json')
    suite = sub.add_parser('experiments', help='六组完整实验、验证集选参及配置封存')
    suite.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    suite.add_argument('--max-users', type=int, default=20)
    suite.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'suite')
    report = sub.add_parser('report', help='依据已保存实验生成图表及设计报告')
    report.add_argument('--suite', type=Path, default=ROOT / 'outputs' / 'suite')
    args = parser.parse_args()
    try:
        if args.command == 'generate':
            result = generate(args.data, seed=args.seed)
            save_json(ROOT / 'outputs' / 'generation.json', result)
            print(json.dumps(result, ensure_ascii=False))
            return
        if args.command == 'report':
            from recsys.reporting import build_report
            build_report(args.suite, ROOT)
            return
        engine = Engine(args.data)
        if args.command == 'experiments':
            from recsys.experiments import run_suite
            run_suite(engine, args.output, max_users=args.max_users,
                      seeds=[int(seed) for seed in args.seeds.split(',')])
            return
        if args.command == 'evaluate':
            evaluate(engine, [int(seed) for seed in args.seeds.split(',')], args.max_users, args.output)
            return
        result = engine.recommend(args.user, method=args.method, k=args.k,
                                  config=Config(seed=args.seed), m=args.candidates,
                                  price_min=args.price_min, price_max=args.price_max)
        save_json(args.output, result)
        save_json(ROOT / 'outputs' / 'data_audit.json', engine.stats)
        for item in result['items']:
            print(f'{item["rank"]:2d}. {item.get("name", item["item_id"])} '
                  f'￥{item["price"]:.2f}  {item["reason"]}')
        print(f'{result["status"]} | R={result["R"]:.4f} D={result["D"]:.4f} '
              f'V={result["V"]:.4f} F={result["F"]:.4f} '
              f'耗时={result["request_seconds"]:.3f}s')
        print(f'结果与收敛历史保存至 {args.output}')
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'错误：{error}\n')


if __name__ == '__main__':
    main()

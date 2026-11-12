"""Create standalone experiment figures and a report from saved results only."""
import base64
import html
import json
import re
from pathlib import Path

import numpy as np


LABELS = {'popular': '热门推荐', 'greedy': '相关性贪心',
          'ga_default': '遗传算法 默认', 'ga_locked': '遗传算法 验证选参', 'ga': '遗传算法'}
METRICS = [('ndcg', 'NDCG@10'), ('D', '多样性 D'), ('V', '新颖性 V'), ('coverage', '目录覆盖率')]


def metric(variant, name, part='mean'):
    value = variant.get('summary', {}).get(name, {})
    return float(value.get(part, 0)) if isinstance(value, dict) else float(value)


def label(variant):
    return LABELS.get(variant['name'], variant['name'])


def figures(suite, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.sans-serif': [name for name in
                         ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'] if name in installed],
                         'axes.unicode_minus': False, 'figure.dpi': 130,
                         'savefig.dpi': 180, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlepad': 14,
                         'axes.grid': True, 'grid.alpha': .17, 'axes.axisbelow': True,
                         'font.size': 10, 'figure.facecolor': '#ffffff'})
    folder.mkdir(parents=True, exist_ok=True)
    colors = ['#8291a8', '#1e9d8b', '#ea8a42', '#4267c5']

    def finish(fig, name):
        fig.savefig(folder / f'{name}.png', bbox_inches='tight')
        fig.savefig(folder / f'{name}.svg', bbox_inches='tight')
        plt.close(fig)

    baseline = suite['experiments']['baseline']
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), layout='constrained')
    for ax, (key, title) in zip(axes.flat, METRICS):
        values = [metric(row, key) for row in baseline]
        errors = [metric(row, key, 'std') for row in baseline]
        ax.bar(np.arange(len(baseline)), values, yerr=errors, capsize=3,
               color=colors[:len(baseline)], width=.62)
        ax.set_xticks(np.arange(len(baseline)), [label(row).replace(' ', '\n') for row in baseline])
        ax.set_title(title)
        ax.set_ylim(0, max(.12, max(values) * 1.28))
        for index, value in enumerate(values):
            ax.text(index, value + max(.003, max(values) * .04), f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9)
    fig.suptitle('测试集算法对比 · 误差线为种子间样本标准差', fontsize=14)
    finish(fig, 'algorithm_comparison')

    cases = suite.get('cases', [])
    fig, axes = plt.subplots(1, max(1, len(cases)), figsize=(12, 3.8), squeeze=False,
                             layout='constrained')
    for ax, case in zip(axes.flat, cases):
        result = case['results'].get('ga', {})
        history = result.get('history', [])
        if history:
            ax.plot([x['generation'] for x in history], [x['best'] for x in history],
                    color=colors[3], label='历史最优 F', linewidth=2)
            ax.plot([x['generation'] for x in history], [x['mean'] for x in history],
                    color=colors[0], label='种群平均 F')
        ax.set(title=f'用户 {case["user_id"]}', xlabel='代数', ylabel='适应度 F')
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('不同用户的遗传算法收敛过程', fontsize=14)
    finish(fig, 'convergence')

    weights = suite['experiments']['weights']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout='constrained')
    for index, variant in enumerate(weights):
        weight = variant['config']['weights']
        text = '/'.join(f'{x:g}' for x in weight)
        for ax, y, title in zip(axes, ['R', 'ndcg', 'V'],
                                ['相关性与多样性', '排序质量与多样性', '新颖性与多样性']):
            ax.errorbar(metric(variant, 'D'), metric(variant, y),
                        xerr=metric(variant, 'D', 'std'), yerr=metric(variant, y, 'std'),
                        fmt='o', color=colors[index % 4], capsize=3, markersize=8, label=text)
            ax.set(xlabel='多样性 D', ylabel={'R':'相关性 R', 'ndcg':'NDCG@10', 'V':'新颖性 V'}[y], title=title)
    axes[0].legend(title='R / D / V 权重', frameon=False)
    fig.suptitle('验证集目标权重取舍 · 固定候选和计算配置', fontsize=14)
    finish(fig, 'weight_tradeoff')

    params = suite['experiments']['parameters']
    default = next((row for row in params if row['config']['population'] == 60
                    and row['config']['crossover'] == .8 and row['config']['mutation'] == .2), params[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout='constrained')
    for ax, key, title in zip(axes, ['population', 'crossover', 'mutation'],
                              ['种群大小 N', '交叉概率 Pc', '变异概率 Pm']):
        others = [x for x in ['population', 'crossover', 'mutation'] if x != key]
        subset = [row for row in params if all(row['config'][x] == default['config'][x] for x in others)]
        subset = sorted(subset, key=lambda row: row['config'][key])
        ax.errorbar([row['config'][key] for row in subset], [metric(row, 'ndcg') for row in subset],
                    yerr=[metric(row, 'ndcg', 'std') for row in subset], color=colors[3],
                    marker='o', capsize=3)
        ax.set(xlabel=title, ylabel='验证 NDCG@10')
    fig.suptitle('单因素参数敏感性 · 其余参数固定为默认值', fontsize=14)
    finish(fig, 'parameter_sensitivity')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, key, title in zip(axes, ['population', 'crossover', 'mutation'],
                              ['种群大小 N', '交叉概率 Pc', '变异概率 Pm']):
        others = [x for x in ['population', 'crossover', 'mutation'] if x != key]
        subset = sorted([row for row in params if all(row['config'][x] == default['config'][x]
                         for x in others)], key=lambda row: row['config'][key])
        for index, row in enumerate(subset):
            history = row.get('convergence', [])
            color = colors[index % 4]
            ax.plot([x['generation'] for x in history], [x['best'] for x in history],
                    color=color, label=f'{row["config"][key]:g} 最优')
            ax.plot([x['generation'] for x in history], [x['mean'] for x in history],
                    color=color, linestyle='--', alpha=.65)
        ax.set(xlabel='代数', ylabel='适应度 F', title=title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('参数组收敛过程 · 实线最优，虚线平均 · 同一用户与种子', fontsize=13)
    finish(fig, 'parameter_convergence')

    scale = sorted(suite['experiments']['scale'], key=lambda row: row['m'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot([row['m'] for row in scale],
                 [row.get('optimization_p95_seconds', 0) for row in scale],
                 marker='o', color=colors[3], label='优化 P95')
    axes[0].plot([row['m'] for row in scale], [row.get('request_p95_seconds', 0) for row in scale],
                 marker='s', color=colors[1], label='含候选生成 P95')
    axes[0].set(xlabel='候选数 M', ylabel='耗时 / 秒', title='规模与运行时间')
    axes[0].legend(frameon=False)
    axes[1].plot([row['m'] for row in scale], [metric(row, 'candidate_recall') for row in scale],
                 marker='o', color=colors[2])
    axes[1].set(xlabel='候选数 M', ylabel='候选召回率', ylim=(0, 1), title='候选池对可命中的限制')
    finish(fig, 'scale')


def table(headers, rows):
    def clean(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(clean, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows])


def markdown_html(markdown, directory):
    """Small deterministic renderer for this generated report, embedding local figures."""
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        return text

    output, lines, index = [], markdown.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        if line.startswith('```'):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith('```'):
                code.append(lines[index])
                index += 1
            output.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
        elif line.startswith('|'):
            rows = []
            while index < len(lines) and lines[index].startswith('|'):
                cells = [cell.strip() for cell in lines[index].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                    rows.append(cells)
                index += 1
            output.append('<table><thead><tr>' + ''.join('<th>' + inline(x) + '</th>' for x in rows[0]) +
                          '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + inline(x) + '</td>'
                          for x in row) + '</tr>' for row in rows[1:]) + '</tbody></table>')
            continue
        elif re.match(r'!\[.*\]\(.*\)', line):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
            path = directory / match.group(2)
            if path.exists():
                data = base64.b64encode(path.read_bytes()).decode('ascii')
                output.append(f'<figure><img src="data:image/png;base64,{data}" alt="'
                              f'{html.escape(match.group(1))}"><figcaption>'
                              f'{inline(match.group(1))}</figcaption></figure>')
        elif line.startswith('#'):
            level = min(6, len(line) - len(line.lstrip('#')))
            output.append(f'<h{level}>' + inline(line[level:].strip()) + f'</h{level}>')
        elif line.startswith('- '):
            output.append('<p class="bullet">• ' + inline(line[2:]) + '</p>')
        elif line.strip():
            output.append('<p>' + inline(line) + '</p>')
        index += 1
    style = '''body{font:15px/1.85 "Microsoft YaHei",sans-serif;color:#263247;background:#edf1f6;margin:0}
main{max-width:1000px;margin:32px auto;padding:48px 60px;background:white;box-shadow:0 8px 40px #23334b12}
h1{font-size:29px;color:#121c2d;margin:0 0 14px}h2{font-size:21px;color:#121c2d;margin-top:38px}
h3{font-size:17px;margin-top:26px}p{margin:12px 0}table{width:100%;border-collapse:collapse;font-size:13px;margin:20px 0}
th,td{border:1px solid #d9dfe7;padding:9px 11px;text-align:left}th{background:#f0f4fa;color:#14253e}
tbody tr:nth-child(even){background:#fafbfd}img{max-width:100%;height:auto}figure{margin:24px 0;break-inside:avoid}
figcaption{text-align:center;color:#64748b;font-size:12px}code{font-family:Consolas,monospace;background:#f2f5f8;padding:2px 4px}
pre{white-space:pre-wrap;background:#f2f5f8;padding:18px;border-radius:6px;font-size:12px}code{overflow-wrap:anywhere}
@media print{body{background:white}main{margin:0;padding:0;box-shadow:none;max-width:none}
@page{size:A4;margin:18mm}h2,h3{break-after:avoid}tr{break-inside:avoid}p{orphans:3;widows:3}}'''
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>网购平台物品推送系统设计报告</title>' + \
           '<style>' + style + '</style><main>' + '\n'.join(output) + '</main></html>'


def build_report(suite_dir, project_dir):
    suite_dir, project_dir = Path(suite_dir), Path(project_dir)
    suite = json.loads((suite_dir / 'suite.json').read_text(encoding='utf-8'))
    if not suite.get('meta', {}).get('complete'):
        raise ValueError('完整实验尚未完成，不能生成最终报告')
    docs = project_dir / 'docs'
    docs.mkdir(parents=True, exist_ok=True)
    figures(suite, docs / 'figures')
    groups, cohort = suite['experiments'], suite['cohort']
    baseline = groups['baseline']
    locked = suite['locked_config']
    from .data import write_csv
    write_csv(suite_dir / 'evaluation_users.csv', [{'user_id': uid} for uid in cohort['selected_users']],
              ['user_id'])
    stats = suite['meta']['data_split']
    raw_rows = []
    for group, variants in groups.items():
        if group == 'edge':
            continue
        for variant in variants:
            raw = json.loads((suite_dir / variant['per_user_file']).read_text(encoding='utf-8'))
            for record in raw['per_user']:
                row = dict(group=group, variant=variant['name'], split=variant['split'],
                           method=variant['method'])
                for key in ['user_id','seed','precision','recall','ndcg','R','D','V','F',
                            'candidate_recall','candidate_count','positive_count','returned',
                            'generations','optimization_seconds','request_seconds']:
                    row[key] = record[key]
                row['item_ids'] = json.dumps(record['item_ids'])
                raw_rows.append(row)
    write_csv(suite_dir / 'per_user_metrics.csv', raw_rows, list(raw_rows[0]))
    sections = ['# 网购平台物品推送系统设计报告',
                '人工智能与模式识别课程项目设计',
                '姓名：________　学号：________　班级：________',
                '## 1 设计任务与系统范围',
                '本项目实现面向“猜你喜欢”的离线商品推荐系统。根据浏览、收藏、购买建立用户画像，'
                '通过候选筛选和遗传算法同时优化商品选择与展示顺序，并比较相关性、多样性、新颖性和运行开销。'
                '本地桌面界面提供用户与算法选择、价格区间、遗传参数调整、推荐解释和收敛曲线。',
                '实验显示，综合适应度提高与排序准确性提高并不等价。本报告保留原始指标，'
                '并将验证集选参、测试集评价与异常验证分开记录。',
                '## 2 数据生成与时间协议',
                table(['项目', '实际记录'], [
                    ['用户 / 商品 / 类别', f'{stats["users"]} / {stats["items"]} / {stats["known_categories"]}'],
                    ['原始 / 清洗后行为', f'{stats["raw_records"]} / {stats["clean_records"]}'],
                    ['训练 / 验证 / 测试行为', f'{stats["train_records"]} / {stats["validation_records"]} / {stats["test_records"]}'],
                    ['验证起点', stats['validation_start']], ['测试起点', stats['test_start']],
                    ['筛选前 / 合格 / 实际评价用户', f'{cohort["users_before_filter"]} / {cohort["eligible_count"]} / {len(cohort["selected_users"])}']]),
                '模拟生成种子为42。用户具有潜在类别偏好与价格偏好，商品具有随机吸引力，交互由这些因素和噪声共同产生。'
                '潜在偏好不直接提供给推荐器，生成器不调用推荐适应度产生标签。所有商品属性从模拟开始已知且静态；'
                '因此已知商品评分中位数等统计不包含后续反馈。数据规模和模拟假设仅用于教学仿真。',
                'CSV包含users、items、behaviors三张表。浏览按用户、商品、分钟合并；收藏和购买只去除完全一致的事件。'
                '非法关联、动作和时间丢弃，商品编号重复或非法属性报错。缺失类别设为未知，未知不计入类别数量；'
                '缺失评分以训练期已知静态商品评分中位数填充，全部缺失时质量分为0.5。',
                '行为按北京时间全局排序，以约70%/15%/15%划分，分界不拆开同一时间戳。画像、价格偏好和热度固定由训练集产生。'
                '已购过滤使用请求起点前已知的购买记录。主实验用户训练行为至少10条，两个评价区间均有有效收藏或购买，'
                '且两次候选数均至少10；按用户编号升序取前20名。该确定性选择可复现，但不代表全体用户的无偏抽样。',
                '## 3 用户画像与候选生成',
                '商品使用类别独热向量并进行L2归一化。用户画像为训练期商品向量的加权和，再L2归一化；'
                '浏览、收藏、购买权重分别为1、3、5。这些权重属于教学假设，不表示真实购买概率。',
                '`s(u,i) = 0.60 × cos(h_u,x_i) + 0.25 × b(u,i) + 0.15 × q_i`',
                '`b(u,i) = 1 / (1 + abs(price_i - median_u) / (median_u + 0.000001))`',
                '质量分q为评分除以5。用户价格中心优先使用训练购买价格中位数，其次浏览/收藏价格，再其次训练期已知商品价格。'
                '先过滤下架、零库存、已购、屏蔽和超出单件价格范围的商品，得到可推荐全集A。价格范围约束每件商品，'
                '并非整张列表总预算。混合候选先取相关性最高的ceil(0.8M)件，按类别名升序轮流补足M，分数相同按编号升序。',
                '无画像用户在A中按训练热度回退，不运行遗传优化。候选不足K时返回实际长度；空候选有明确提示；'
                '展示前重新检查上架状态与库存。',
                '## 4 遗传算法设计',
                '染色体为K件商品编号的有序、不重复列表。初始种群包含相关性贪心列表，其余个体无放回随机抽取并打乱。'
                '每代保留2个精英，使用规模3的锦标赛选择父代，前缀交叉后去重补全；变异等概率交换位置或替换商品。'
                '所有操作复制父代并修复非法编号及重复项。',
                '`d_k = 1 / log2(k+1)`；R和V分别为位置折扣后的相关性与新颖性加权平均。',
                '`D = 平均(1 - cos(x_a,x_b))`，遍历列表全部不同位置对；K小于2时D=0。',
                '`v_i = 1 - ln(1+c_i) / ln(1+c_max)`，c为训练期去重交互用户数；c_max为0时所有v=1。',
                '`F = w_R × R + w_D × D + w_V × V`；默认权重为0.7、0.2、0.1。',
                '默认种群60，交叉概率0.8，变异概率0.2，最多100代。最优值连续15代的相邻提升不超过0.0001时提前停止。'
                '记录每代平均F和历史最优F。贪心初始化与精英保留保证同一权重下最终F不低于贪心，'
                '但不能由此推导NDCG或Recall改善。不同权重的F不作为同一种效用横向比较。',
                '## 5 实验协议与指标',
                '实验计划先保存manifest.json，验证集执行参数和权重比较，按预定义规则选择并保存locked_config.json，'
                '再执行最终测试比较。2026年11月3日曾查看默认配置的测试结果，因此本轮测试不能宣称是从未见过的全新留出集；'
                '本轮参数选择仅依据验证结果，历史结果保留，未根据测试排序反复挑参。',
                '所有主实验算法共享同一用户、可推荐全集和候选池。热门按训练去重交互用户数，贪心按单件相关性排序。'
                '有效正反馈为评价区间收藏或购买去重集合，并按请求时刻硬约束过滤；真值全集不按候选池截断。',
                'Precision@10=命中数/10，Recall@10=命中数/有效正反馈数。二元NDCG@10使用位置折扣，'
                'IDCG由完整有效真值确定。候选召回率单独报告，目录覆盖率为所有用户推荐并集除以可推荐全集并集。'
                '先对用户宏平均，再对10个种子计算均值与样本标准差。无正反馈的异常用户不混入主实验均值。',
                'GA使用种子0至9，计时前预热。优化计时排除文件加载、图表和界面；含候选生成的请求计时另列。'
                '同种子复现比较商品编号与收敛序列，耗时不要求一致。完整环境、硬件和数据哈希见suite.json与manifest.json。',
                '封存方案：`' + locked['variant_name'] + '`。选择规则：' + locked['rule'],
                '封存配置：`' + json.dumps(locked['config'], ensure_ascii=False, separators=(',', ':')) + '`',
                '## 6 实验结果与分析',
                '### 6.1 测试集算法对比',
                table(['算法', 'Precision@10', 'Recall@10', 'NDCG@10 均值±标准差', 'D', 'V', '覆盖率'],
                      [[label(row), f'{metric(row,"precision"):.4f}', f'{metric(row,"recall"):.4f}',
                        f'{metric(row,"ndcg"):.4f} ± {metric(row,"ndcg","std"):.4f}',
                        f'{metric(row,"D"):.4f}', f'{metric(row,"V"):.4f}',
                        f'{metric(row,"coverage"):.4f}'] for row in baseline]),
                '![测试集算法对比](figures/algorithm_comparison.png)',
                '![三名用户收敛过程](figures/convergence.png)']
    popular = next(row for row in baseline if row['method'] == 'popular')
    for row in baseline:
        if row['method'] == 'ga':
            relation = '达到' if metric(row, 'ndcg') >= metric(popular, 'ndcg') else '未达到'
            sections.append(f'{label(row)}的NDCG@10为{metric(row,"ndcg"):.4f}，热门为'
                            f'{metric(popular,"ndcg"):.4f}，{relation}“NDCG不低于热门”的设计目标。'
                            f'其多样性D为{metric(row,"D"):.4f}，说明应结合准确性与多样性分别判断结果。')
    if locked['config']['weights'] == [1.0, 0.0, 0.0]:
        sections.append('验证集最终选择纯相关性权重(1,0,0)。当目标只包含位置加权相关性时，'
                        '按相关性从高到低取前K件即为最优，初始化已包含该解，因此封存GA与贪心结果一致。'
                        '这说明本轮选参倾向放弃多样性目标，不能将其写为遗传搜索超越贪心。')
    sections.append(table(['测试方案', '优化P95 秒', '含候选生成请求P95 秒'],
                          [[label(row), f'{row.get("optimization_p95_seconds",0):.4f}',
                            f'{row.get("request_p95_seconds",0):.4f}'] for row in baseline]))
    sections += ['### 6.2 验证集参数敏感性',
                 table(['方案', 'N', 'Pc', 'Pm', 'NDCG@10', 'F', '平均代数', '优化P95 秒'],
                       [[row['name'], row['config']['population'], row['config']['crossover'],
                         row['config']['mutation'], f'{metric(row,"ndcg"):.4f}',
                         f'{metric(row,"F"):.4f}', f'{metric(row,"generations"):.1f}',
                         f'{row.get("optimization_p95_seconds",0):.4f}']
                        for row in groups['parameters']]),
                 '![验证集参数敏感性](figures/parameter_sensitivity.png)',
                 '![参数组最优与平均适应度曲线](figures/parameter_convergence.png)',
                 '参数比较按单因素执行。增大种群会扩大每代搜索量，但是否提升排序需要看实测NDCG；'
                 '收敛F、实际停止代数和时间也应共同观察，不能仅以更高F决定准确性。',
                 '### 6.3 目标权重取舍',
                 table(['R/D/V 权重', 'NDCG@10', 'R', 'D', 'V'],
                       [[str(row['config']['weights'])] + [f'{metric(row,key):.4f}'
                         for key in ['ndcg', 'R', 'D', 'V']] for row in groups['weights']]),
                 '![目标权重取舍](figures/weight_tradeoff.png)',
                 '目标函数直接奖励类别差异和低热度物品。模拟交互还受商品吸引力与噪声影响，而当前模型只使用类别、价格和评分；'
                 '这一模型与行为机制的差异是排序落后时需要检验的假设，不能仅凭本轮结果认定因果。',
                 '### 6.4 冷启动与异常',
                 table(['场景', '检查结果', '说明'], [[row['name'], '通过' if row.get('passed') else '未通过',
                       row.get('notes') or str(row.get('details', ''))] for row in groups['edge']]),
                 '异常结果单独汇总，不计入主实验准确率。库存变动为请求过程中的受控测试，不代表已实现历史商品快照。',
                 '### 6.5 消融与候选上限',
                 table(['方案', 'NDCG@10', 'D', '候选召回率', 'F'],
                       [[row['name']] + [f'{metric(row,key):.4f}' for key in ['ndcg','D','candidate_recall','F']]
                        for row in groups['ablation']]),
                 '仅交换变异不能从该次变异中引入未选商品；交叉仍可重组种群已有编号。'
                 '替换变异提供额外的选品探索。纯相关性候选与混合候选的候选召回及类别分布用于检查上游限制，'
                 '候选未召回的正反馈无法仅靠增加遗传代数恢复。',
                 '### 6.6 规模与可复现性',
                 table(['候选 M', 'NDCG@10', 'F 均值±标准差', '平均代数', '候选召回率', '优化P95 秒', '请求P95 秒'],
                       [[row['m'], f'{metric(row,"ndcg"):.4f}',
                         f'{metric(row,"F"):.4f} ± {metric(row,"F","std"):.4f}',
                         f'{metric(row,"generations"):.1f}',
                         f'{metric(row,"candidate_recall"):.4f}',
                         f'{row.get("optimization_p95_seconds",0):.4f}',
                         f'{row.get("request_p95_seconds",0):.4f}'] for row in groups['scale']]),
                 '![候选规模与耗时](figures/scale.png)',
                 '固定候选、目标和种子时应复现推荐顺序及收敛历史。实际复跑记录及环境配置保存在实验JSON。'
                 'P95不超过2秒为本机优化阶段的目标，结论仅适用于所记录的硬件与实现。',
                 '## 7 用户案例与系统展示']
    for case in suite.get('cases', []):
        sections += [f'### 用户 {case["user_id"]}',
                     '训练画像：`' + json.dumps(case['profile'], ensure_ascii=False) + '`',
                     '候选类别分布：`' + json.dumps(case.get('candidate_categories', {}), ensure_ascii=False) + '`']
        result = case['results']['ga']
        sections.append(table(['位置', '商品编号', '类别', '价格', '相关性', '理由'],
                              [[x['rank'], x['item_id'], x['category'], f'{x["price"]:.2f}',
                                f'{x["relevance"]:.4f}', x['reason']] for x in result['items']]))
    screenshot = project_dir / 'outputs' / 'screenshots' / 'demo.png'
    if screenshot.exists():
        sections += ['![本地桌面推荐演示界面](../outputs/screenshots/demo.png)']
    if (project_dir / 'outputs' / 'screenshots' / 'profile.png').exists():
        sections += ['![用户画像与候选类别分布](../outputs/screenshots/profile.png)']
    sections += ['## 8 局限与改进',
                 '当前类别独热特征无法区分同类商品的细粒度兴趣，评分与行为机制也未必一致。'
                 '可以在验证协议不变的条件下加入训练期已知标签特征或协同候选；新品探索和时间衰减应另立对照。'
                 '模拟数据、少量确定性选择的评价用户、已经看过的历史测试结果均限制了结果外推。'
                 '下一阶段如继续研究，应预先规划新的数据或独立留出集，而非反复依据当前测试成绩修改方案。',
                 '课程代码和实验报告已给出运行方法及实际结果。姓名、学号、班级与小组分工应由实际参与者按真实情况填写。',
                 '## 9 运行与提交材料',
                 '```powershell\npython -m pip install -r requirements.txt\n'
                 'python app.py\npython -m unittest discover -s tests -v\n'
                 'python tests/ui_smoke.py\npython main.py experiments\npython main.py report\n```',
                 '已有数据保存在data中；仅在需要重建时运行python main.py generate。'
                 '完整实验结果位于outputs/suite，包含封存计划、配置、CSV汇总、逐用户原始结果、复现和异常记录；'
                 'per_user_metrics.csv可直接用于表格分析，保留组别、方案、用户、种子和推荐编号。'
                 '清洗后行为和评价用户清单分别为behaviors_clean.csv、evaluation_users.csv。'
                 '图表位于docs/figures，同时提供PNG与SVG。HTML报告嵌入图片，可独立打开并打印。',
                 '设计依据为references/项目设计指导书.docx。未接入真实用户、支付或在线通知服务。']
    content = '\n\n'.join(sections) + '\n'
    target = docs / '项目设计报告.md'
    target.write_text(content, encoding='utf-8')
    (docs / '项目设计报告.html').write_text(markdown_html(content, docs), encoding='utf-8')
    print(f'图表与报告已生成：{target}', flush=True)

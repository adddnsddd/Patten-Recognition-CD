"""Checks performed inside the standalone executable, without external Python."""
import json
from pathlib import Path
import sys
import time
import traceback


def check(app, output):
    destination = Path(output).resolve()
    report = dict(passed=False, frozen=bool(getattr(sys, 'frozen', False)),
                  executable=sys.executable, bundled_data=str(app.data_folder), checks=[])

    def generate():
        previous = app.completed_jobs
        app.generate()
        deadline = time.monotonic() + 30
        while app.completed_jobs == previous:
            if app.last_error:
                raise AssertionError(app.last_error)
            if time.monotonic() > deadline:
                raise AssertionError('推荐任务超时')
            app.root.update()
            time.sleep(.01)
        app.root.update()
        if app.last_error is not None:
            raise RuntimeError(app.last_error)
        return app.result

    try:
        assert app.engine is not None, app.last_error
        app.root.update()
        report['checks'].append('内置数据读取与Tk窗口创建')
        result = generate()
        assert result['returned'] == 10
        assert len(app.tree.get_children()) == 10
        assert len(app.native_charts.convergence_canvas.find_all()) > 8
        assert result['F'] + 1e-12 >= app.engine.recommend(1, 'greedy')['F']
        first_ids = [item['item_id'] for item in result['items']]
        first_history = result['history']
        report['checks'].append('遗传推荐、精英保持与原生收敛曲线')
        again = generate()
        assert first_ids == [item['item_id'] for item in again['items']]
        assert first_history == again['history']
        report['checks'].append('同种子结果与曲线可复现')
        app.chart_notebook.select(1)
        app.root.update()
        assert len(app.native_charts.profile_canvas.find_all()) > 10
        report['checks'].append('原生用户画像及候选分布图')
        app.price_min_var.set('50')
        app.price_max_var.set('300')
        for title, method in [('相关性贪心', 'greedy'), ('热门推荐', 'popular')]:
            app.method_var.set(title)
            result = generate()
            assert result['method'] == method
            assert all(50 <= item['price'] <= 300 for item in result['items'])
        report['checks'].append('三种算法切换及单件价格约束')
        app.user_var.set('9999（新用户）')
        app.method_var.set('遗传算法 GA')
        result = generate()
        assert result['cold_start'] and result['method'] == 'popular'
        assert not result['history']
        report['checks'].append('新用户热门回退')
        app.user_var.set('2')
        app.price_min_var.set('0')
        app.price_max_var.set('0')
        result = generate()
        assert result['returned'] == 0
        assert not app.tree.get_children()
        report['checks'].append('空候选提示')
        export = destination.parent / '推荐结果 中文路径.json'
        app.export_to(export)
        saved = json.loads(export.read_text(encoding='utf-8'))
        assert saved['user_id'] == 2 and saved['returned'] == 0
        report['checks'].append('中文路径JSON导出')
        assert 'matplotlib' not in sys.modules and 'PIL' not in sys.modules
        report['checks'].append('轻量图表运行不依赖Matplotlib或Pillow')
        report['passed'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report['passed']

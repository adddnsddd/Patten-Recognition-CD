"""Real Tk widgets and background jobs; not part of unittest discovery."""
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wait_for(root, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('界面任务超时')
        root.update()
        time.sleep(0.02)
    root.update()


def run_smoke(data_folder=None):
    from app import RecommendationApp, enable_dpi_awareness
    enable_dpi_awareness()
    root = tk.Tk()
    app = RecommendationApp(root, data_folder or ROOT / 'data', auto_start=False)
    checked = []
    try:
        assert app.engine is not None, app.last_error
        app.generate_button.invoke()
        wait_for(root, lambda: app.completed_jobs == 1)
        assert app.last_error is None, app.last_error
        assert app.result['returned'] == 10
        assert len(app.tree.get_children()) == 10
        assert len(app.convergence_ax.lines) == 2
        assert app.convergence_canvas.get_tk_widget().winfo_width() > 300
        assert app.generate_button.winfo_height() > 30
        assert app.export_button.winfo_y() + app.export_button.winfo_height() < app.export_button.master.winfo_height()
        checked.append('遗传推荐、商品表格及收敛图')
        screenshot = ROOT / 'outputs' / 'screenshots' / 'demo.png'
        app.capture(screenshot)

        app.chart_notebook.select(1)
        root.update()
        assert len(app.profile_axes[0].patches) == len(app.result['profile'])
        assert len(app.profile_axes[1].patches) == len(app.result['profile'])
        app.capture(ROOT / 'outputs' / 'screenshots' / 'profile.png')
        checked.append('用户画像与候选类别图')

        app.method_var.set('相关性贪心')
        app.price_min_var.set('50')
        app.price_max_var.set('300')
        app.generate_button.invoke()
        wait_for(root, lambda: app.completed_jobs == 2)
        assert app.last_error is None, app.last_error
        assert app.result['method'] == 'greedy'
        assert all(50 <= item['price'] <= 300 for item in app.result['items'])
        assert not app.convergence_ax.lines
        checked.append('算法切换与单件价格范围')

        app.user_var.set('9999（新用户）')
        app.method_var.set('遗传算法 GA')
        app.generate_button.invoke()
        wait_for(root, lambda: app.completed_jobs == 3)
        assert app.result['cold_start'] and app.result['method'] == 'popular'
        assert '回退' in app.status.get()
        checked.append('新用户热门回退')

        app.user_var.set('2')
        app.price_min_var.set('0')
        app.price_max_var.set('0')
        app.generate_button.invoke()
        wait_for(root, lambda: app.completed_jobs == 4)
        assert app.result['returned'] == 0
        assert not app.tree.get_children()
        assert '暂无' in app.status.get()
        checked.append('空候选与解释提示')

        app.price_min_var.set('500')
        app.price_max_var.set('100')
        app.generate_button.invoke()
        wait_for(root, lambda: app.completed_jobs == 5)
        assert app.last_error is not None
        assert not app.busy
        checked.append('非法参数处理及后台任务结束')

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / '推荐 结果.json'
            app.export_to(target)
            saved = json.loads(target.read_text(encoding='utf-8'))
            assert saved['user_id'] == 2 and saved['returned'] == 0
        checked.append('中文路径JSON导出')
        app.price_min_var.set('50')
        app.price_max_var.set('300')
        app.population_var.set('30')
        app.generations_var.set('20')
        app.seed_var.set('3')
        app.weight_r_var.set('1')
        app.weight_d_var.set('0')
        app.weight_v_var.set('0')
        app.generate_button.invoke()
        wait_for(root, lambda: app.completed_jobs == 6)
        assert app.last_error is None, app.last_error
        assert app.result['config']['population'] == 30
        assert app.result['config']['seed'] == 3
        assert app.result['config']['weights'] == (1., 0., 0.)
        assert len(app.result['history']) <= 21
        checked.append('遗传参数和目标权重可调整')
        report = dict(status='passed', checks=checked, screenshot=str(screenshot))
        target = ROOT / 'outputs' / 'ui_smoke.json'
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))
        return report
    finally:
        app.close()


if __name__ == '__main__':
    run_smoke()

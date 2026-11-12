"""Local desktop demonstration. Run ``python app.py`` from any directory."""
import argparse
import json
from pathlib import Path
import queue
import threading
import time
import sys
import tkinter as tk
from tkinter import filedialog, ttk

LIGHTWEIGHT = bool(getattr(sys, 'frozen', False)) or '--lightweight' in sys.argv
if not LIGHTWEIGHT:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib import font_manager

from recsys.core import Engine
from recsys.ga import Config


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else ROOT / 'outputs'
NAVY = '#142C3A'
TEAL = '#087F8C'
ORANGE = '#E1974B'
BG = '#F2F5F6'
INK = '#203743'
MUTED = '#667D87'
METHODS = {'遗传算法 GA': 'ga', '相关性贪心': 'greedy', '热门推荐': 'popular'}


def enable_dpi_awareness():
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def configure_fonts():
    if LIGHTWEIGHT:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC')
                if name in available]
    matplotlib.rcParams['font.sans-serif'] = families + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 9


class RecommendationApp:
    def __init__(self, root, data_folder=None, auto_start=True):
        self.root = root
        self.closed = False
        self.busy = False
        self.result = None
        self.last_error = None
        self.completed_jobs = 0
        self._queue = queue.Queue()
        self.engine = None
        self.data_folder = Path(data_folder or ROOT / 'data')
        configure_fonts()
        self.root.title('猜你喜欢 · 网购推荐实验室')
        width = min(1400, self.root.winfo_screenwidth() - 80)
        height = min(860, self.root.winfo_screenheight() - 90)
        self.root.geometry(f'{width}x{height}+40+35')
        self.root.minsize(1120, 700)
        self.root.configure(background=BG)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.option_add('*Font', ('Microsoft YaHei UI', 10))
        self._style()
        self._variables()
        self._build()
        try:
            self.engine = Engine(self.data_folder)
            self.user_box['values'] = [str(uid) for uid in sorted(self.engine.users)] + ['9999（新用户）']
            self.dataset_text.set(f'{len(self.engine.users)} 位用户  /  {len(self.engine.items)} 件商品'
                                  f'  /  {len(self.engine.events):,} 条行为')
            self.status.set('数据已就绪。选择用户与算法，生成个性化推荐。')
            if auto_start:
                self.root.after(150, self.generate)
        except (FileNotFoundError, ValueError) as error:
            self.last_error = str(error)
            self.status.set('数据未就绪：请先在项目目录运行 python main.py generate，再重新打开演示。')
            self.generate_button.configure(state='disabled')
            self.dataset_text.set('等待数据 · 不会覆盖现有数据')
        self.root.after(40, self._poll)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10), foreground=INK)
        style.configure('TFrame', background=BG)
        style.configure('White.TFrame', background='white')
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('White.TLabel', background='white', foreground=INK)
        style.configure('Muted.TLabel', background='white', foreground=MUTED, font=('Microsoft YaHei UI', 9))
        style.configure('Title.TLabel', background=BG, font=('Microsoft YaHei UI', 14, 'bold'))
        style.configure('TEntry', fieldbackground='white', padding=(7, 5))
        style.configure('TCombobox', fieldbackground='white', padding=(7, 5))
        style.configure('TButton', padding=(12, 8), background='#E5ECEF', borderwidth=0)
        style.configure('Accent.TButton', background=TEAL, foreground='white', padding=(12, 11),
                        font=('Microsoft YaHei UI', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#066F7A'), ('disabled', '#A5BEC2')])
        style.configure('TNotebook', background='white', borderwidth=0)
        style.configure('TNotebook.Tab', padding=(12, 7), background='#E7EDF0', foreground=MUTED)
        style.map('TNotebook.Tab', background=[('selected', 'white')], foreground=[('selected', TEAL)])
        style.configure('Treeview', background='white', fieldbackground='white',
                        rowheight=25, borderwidth=0, font=('Microsoft YaHei UI', 9))
        style.configure('Treeview.Heading', background='#E8EEF0', foreground=MUTED,
                        font=('Microsoft YaHei UI', 9, 'bold'), padding=(7, 7), relief='flat')
        style.map('Treeview', background=[('selected', '#D5EBED')], foreground=[('selected', INK)])
        style.configure('Horizontal.TProgressbar', background=TEAL, troughcolor='#E1EBED', borderwidth=0)

    def _variables(self):
        self.user_var = tk.StringVar(value='1')
        self.method_var = tk.StringVar(value='遗传算法 GA')
        self.price_min_var = tk.StringVar(value='')
        self.price_max_var = tk.StringVar(value='')
        self.k_var = tk.StringVar(value='10')
        self.m_var = tk.StringVar(value='100')
        self.seed_var = tk.StringVar(value='0')
        self.population_var = tk.StringVar(value='60')
        self.generations_var = tk.StringVar(value='100')
        self.pc_var = tk.StringVar(value='0.8')
        self.pm_var = tk.StringVar(value='0.2')
        self.weight_r_var = tk.StringVar(value='0.7')
        self.weight_d_var = tk.StringVar(value='0.2')
        self.weight_v_var = tk.StringVar(value='0.1')
        self.status = tk.StringVar(value='正在读取本地数据…')
        self.dataset_text = tk.StringVar(value='正在读取数据')
        self.result_title = tk.StringVar(value='为你挑选，兼顾兴趣与发现')
        self.result_subtitle = tk.StringVar(value='从用户历史出发，观察不同算法如何选择与排列商品。')
        self.metrics = {name: tk.StringVar(value='—') for name in ('R', 'D', 'V', 'F', 'time', 'count')}

    def _build(self):
        header = tk.Frame(self.root, background=NAVY, height=68)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text='猜你喜欢', background=NAVY, foreground='white',
                 font=('Microsoft YaHei UI', 20, 'bold')).pack(side='left', padx=(22, 16))
        tk.Label(header, text='推荐实验室  /  遗传算法课程项目', background=NAVY,
                 foreground='#ABC2CD', font=('Microsoft YaHei UI', 10)).pack(side='left')
        tk.Label(header, textvariable=self.dataset_text, background=NAVY,
                 foreground='#D4E4EA', font=('Microsoft YaHei UI', 9)).pack(side='right', padx=22)

        footer = tk.Frame(self.root, background='#E3EBEE')
        footer.pack(fill='x', side='bottom')
        tk.Label(footer, textvariable=self.status, background='#E3EBEE', foreground=INK,
                 anchor='w', font=('Microsoft YaHei UI', 9)).pack(fill='x', padx=16, pady=7)
        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, padx=16, pady=14)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, style='White.TFrame', width=260)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)
        ttk.Label(sidebar, text='推荐控制台', style='White.TLabel',
                  font=('Microsoft YaHei UI', 12, 'bold')).grid(row=0, column=0, sticky='w', padx=16, pady=(16, 12))
        self.settings_notebook = ttk.Notebook(sidebar)
        self.settings_notebook.grid(row=1, column=0, sticky='nsew', padx=12)
        basic_tab = ttk.Frame(self.settings_notebook, style='White.TFrame')
        parameter_tab = ttk.Frame(self.settings_notebook, style='White.TFrame')
        self.settings_notebook.add(basic_tab, text='推荐设置')
        self.settings_notebook.add(parameter_tab, text='遗传参数')
        basic = self._scrollable(basic_tab)
        parameters = self._scrollable(parameter_tab)

        self._label(basic, '用户')
        self.user_box = ttk.Combobox(basic, textvariable=self.user_var, width=18)
        self.user_box.pack(fill='x', pady=(4, 13))
        self._label(basic, '推荐方法')
        self.method_box = ttk.Combobox(basic, textvariable=self.method_var, values=list(METHODS),
                                      state='readonly', width=18)
        self.method_box.pack(fill='x', pady=(4, 13))
        self._label(basic, '单件商品价格范围 / 元')
        row = ttk.Frame(basic, style='White.TFrame')
        row.pack(fill='x', pady=(4, 4))
        row.columnconfigure((0, 2), weight=1)
        ttk.Entry(row, textvariable=self.price_min_var, width=8).grid(row=0, column=0, sticky='ew')
        ttk.Label(row, text='至', style='White.TLabel').grid(row=0, column=1, padx=6)
        ttk.Entry(row, textvariable=self.price_max_var, width=8).grid(row=0, column=2, sticky='ew')
        ttk.Label(basic, text='留空表示不限制该方向价格', style='Muted.TLabel').pack(anchor='w', pady=(0, 12))
        self._pair(basic, '推荐件数 K', self.k_var, '候选件数 M', self.m_var)
        self._field(basic, '随机种子', self.seed_var)

        self._pair(parameters, '种群规模 N', self.population_var, '最多迭代', self.generations_var)
        self._pair(parameters, '交叉概率 Pc', self.pc_var, '变异概率 Pm', self.pm_var)
        ttk.Label(parameters, text='目标权重（合计为 1）', style='White.TLabel').pack(anchor='w', pady=(8, 5))
        for label, variable in [('相关性 R', self.weight_r_var), ('多样性 D', self.weight_d_var),
                                ('新颖性 V', self.weight_v_var)]:
            self._field(parameters, label, variable)
        ttk.Label(parameters, text='锦标赛 3 / 精英 2\n连续 15 代改善很小时停止',
                  style='Muted.TLabel', justify='left').pack(anchor='w', pady=(10, 2))
        button_area = ttk.Frame(sidebar, style='White.TFrame', padding=(16, 12))
        button_area.grid(row=2, column=0, sticky='ew')
        self.generate_button = ttk.Button(button_area, text='生成推荐  →', style='Accent.TButton', command=self.generate)
        self.generate_button.pack(fill='x')
        self.export_button = ttk.Button(button_area, text='导出本次推荐', command=self.export_dialog, state='disabled')
        self.export_button.pack(fill='x', pady=(8, 0))
        self.progress = ttk.Progressbar(button_area, mode='determinate', length=100)
        self.progress.pack(fill='x', pady=(10, 0))

        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=5)
        main.rowconfigure(3, weight=1)
        title_area = ttk.Frame(main)
        title_area.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        ttk.Label(title_area, textvariable=self.result_title, style='Title.TLabel').pack(anchor='w')
        ttk.Label(title_area, textvariable=self.result_subtitle, foreground=MUTED,
                  font=('Microsoft YaHei UI', 9)).pack(anchor='w', pady=(3, 0))
        self._metric_cards(main)
        table_frame = ttk.Frame(main, style='White.TFrame')
        table_frame.grid(row=2, column=0, sticky='nsew', pady=(10, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ('rank', 'id', 'name', 'category', 'price', 'relevance', 'reason')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10, selectmode='browse')
        specs = [('rank', '序号', 44, 'center'), ('id', '商品编号', 75, 'center'),
                 ('name', '商品', 165, 'w'), ('category', '类别', 65, 'center'),
                 ('price', '价格 / 元', 92, 'e'), ('relevance', '相关性', 72, 'center'),
                 ('reason', '推荐理由', 200, 'w')]
        for column, label, width, anchor in specs:
            self.tree.heading(column, text=label, anchor=anchor)
            self.tree.column(column, width=width, minwidth=width, anchor=anchor,
                             stretch=column in ('name', 'reason'))
        self.tree.tag_configure('odd', background='#F5F8F9')
        self.tree.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=vertical.set)

        self.chart_notebook = ttk.Notebook(main)
        self.chart_notebook.grid(row=3, column=0, sticky='nsew')
        convergence_frame = ttk.Frame(self.chart_notebook, style='White.TFrame')
        profile_frame = ttk.Frame(self.chart_notebook, style='White.TFrame')
        self.chart_notebook.add(convergence_frame, text='迭代收敛')
        self.chart_notebook.add(profile_frame, text='用户画像与候选分布')
        if LIGHTWEIGHT:
            from recsys.tk_charts import NativeCharts
            self.native_charts = NativeCharts(convergence_frame, profile_frame)
        else:
            self.convergence_figure = Figure(figsize=(8.6, 1.55), dpi=100, facecolor='white')
            self.convergence_ax = self.convergence_figure.add_subplot(111)
            self.convergence_canvas = FigureCanvasTkAgg(self.convergence_figure, master=convergence_frame)
            self.convergence_canvas.get_tk_widget().pack(fill='both', expand=True)
            self.profile_figure = Figure(figsize=(8.6, 1.55), dpi=100, facecolor='white')
            self.profile_axes = self.profile_figure.subplots(1, 2)
            self.profile_canvas = FigureCanvasTkAgg(self.profile_figure, master=profile_frame)
            self.profile_canvas.get_tk_widget().pack(fill='both', expand=True)
        self._draw_charts(None)

    @staticmethod
    def _label(parent, text):
        ttk.Label(parent, text=text, style='White.TLabel').pack(anchor='w')

    def _scrollable(self, parent):
        canvas = tk.Canvas(parent, background='white', borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        content = ttk.Frame(canvas, style='White.TFrame', padding=(8, 12))
        window = canvas.create_window(0, 0, window=content, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        content.bind('<Configure>', lambda _event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))

        def wheel(event):
            if not canvas.winfo_viewable():
                return
            x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
            if x <= event.x_root < x + canvas.winfo_width() and y <= event.y_root < y + canvas.winfo_height():
                canvas.yview_scroll(-int(event.delta / 120), 'units')

        self.root.bind('<MouseWheel>', wheel, add='+')
        return content

    def _field(self, parent, label, variable):
        self._label(parent, label)
        ttk.Entry(parent, textvariable=variable, width=18).pack(fill='x', pady=(4, 10))

    def _pair(self, parent, first_label, first, second_label, second):
        frame = ttk.Frame(parent, style='White.TFrame')
        frame.pack(fill='x', pady=(0, 12))
        frame.columnconfigure((0, 1), weight=1)
        for index, (label, variable) in enumerate(((first_label, first), (second_label, second))):
            ttk.Label(frame, text=label, style='White.TLabel').grid(row=0, column=index, sticky='w', padx=(0, 5))
            ttk.Entry(frame, textvariable=variable, width=9).grid(row=1, column=index,
                                                               sticky='ew', padx=(0, 5), pady=(4, 0))

    def _metric_cards(self, parent):
        strip = ttk.Frame(parent)
        strip.grid(row=1, column=0, sticky='ew')
        labels = [('R', '相关性 R'), ('D', '多样性 D'), ('V', '新颖性 V'),
                  ('F', '综合得分 F'), ('time', '计算耗时'), ('count', '候选 / 返回')]
        for index, (name, title) in enumerate(labels):
            strip.columnconfigure(index, weight=1)
            frame = ttk.Frame(strip, style='White.TFrame', padding=(12, 9))
            frame.grid(row=0, column=index, sticky='nsew', padx=(0, 7 if index < 5 else 0))
            ttk.Label(frame, text=title, style='Muted.TLabel').pack(anchor='w')
            ttk.Label(frame, textvariable=self.metrics[name], style='White.TLabel',
                      foreground=TEAL if name == 'F' else INK,
                      font=('Microsoft YaHei UI', 16 if name not in ('count', 'time') else 13, 'bold')).pack(anchor='w')

    def _request(self):
        user = int(self.user_var.get().split('（')[0].strip())
        config = Config(population=int(self.population_var.get()), generations=int(self.generations_var.get()),
                        crossover=float(self.pc_var.get()), mutation=float(self.pm_var.get()),
                        seed=int(self.seed_var.get()), weights=(float(self.weight_r_var.get()),
                        float(self.weight_d_var.get()), float(self.weight_v_var.get()))).validate()
        k, m = int(self.k_var.get()), int(self.m_var.get())
        if not 0 <= k <= 100 or not 0 <= m <= 500:
            raise ValueError('演示支持推荐件数 0–100、候选件数 0–500')
        low = float(self.price_min_var.get()) if self.price_min_var.get().strip() else None
        high = float(self.price_max_var.get()) if self.price_max_var.get().strip() else None
        return user, METHODS[self.method_var.get()], config, dict(k=k, m=m, price_min=low, price_max=high)

    def generate(self):
        if self.busy or self.engine is None:
            return
        try:
            request = self._request()
        except (ValueError, KeyError) as error:
            self.last_error = str(error)
            self.status.set('参数有误：' + str(error))
            return
        self.last_error = None
        self.busy = True
        self.generate_button.configure(state='disabled')
        self.export_button.configure(state='disabled')
        self.progress.configure(mode='indeterminate')
        self.progress.start(14)
        self.status.set('正在生成推荐与计算图表，请稍候…')

        def worker():
            user, method, config, options = request
            try:
                result = self.engine.recommend(user, method=method, config=config, **options)
                self._queue.put((result, None))
            except Exception as error:
                self._queue.put((None, str(error)))

        threading.Thread(target=worker, name='recommendation', daemon=True).start()

    def _poll(self):
        if self.closed:
            return
        try:
            result, error = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.completed_jobs += 1
            self.progress.stop()
            self.progress.configure(mode='determinate', value=0)
            self.generate_button.configure(state='normal')
            if error:
                self.last_error = error
                self.status.set('生成失败：' + error)
                if self.result is not None:
                    self.export_button.configure(state='normal')
            else:
                self.result = result
                self._show(result)
                self.export_button.configure(state='normal')
        self.root.after(40, self._poll)

    def _show(self, result):
        self.tree.delete(*self.tree.get_children())
        for item in result['items']:
            self.tree.insert('', 'end', values=(f'{item["rank"]:02d}', item['item_id'],
                             item.get('name', str(item['item_id'])), item['category'],
                             f'{item["price"]:.2f}', f'{item["relevance"]:.3f}', item['reason']),
                             tags=('odd',) if item['rank'] % 2 else ())
        self.result_title.set(f'为用户 {result["user_id"]:03d} 挑选的 {result["returned"]} 件商品')
        label = next(name for name, method in METHODS.items() if method == result['method'])
        if result['cold_start']:
            self.result_subtitle.set('暂无历史画像，已按训练期热度推荐符合价格与库存条件的商品。')
        elif result['returned'] == 0:
            self.result_subtitle.set('暂无符合条件的商品。可调整价格范围或候选数量，再次生成推荐。')
        else:
            self.result_subtitle.set(f'{label}  ·  价格偏好中位数 ￥{result["median_price"]:.2f}'
                                     f'  ·  请求时刻 {result["requested_at"]}')
        for name in ('R', 'D', 'V', 'F'):
            self.metrics[name].set(f'{result[name]:.3f}')
        self.metrics['time'].set(f'{result["request_seconds"] * 1000:.0f} ms')
        self.metrics['count'].set(f'{result["candidate_count"]} / {result["returned"]}')
        self._draw_charts(result)
        note = '冷启动：已回退热门推荐。' if result['cold_start'] else f'{label}已完成。'
        note += result.get('length_note', '')
        if not result['items']:
            note = result['status']
        self.status.set(note + '  可在图表页查看收敛过程、用户偏好和候选分布，或导出本次结果。')

    @staticmethod
    def _clean_axes(axis):
        axis.set_facecolor('white')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_color('#CFDBDF')
        axis.spines['bottom'].set_color('#CFDBDF')
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.grid(axis='y', color='#E5ECEF', linewidth=0.7)
        axis.set_axisbelow(True)

    def _draw_charts(self, result):
        if LIGHTWEIGHT:
            self.native_charts.draw(result)
            return
        axis = self.convergence_ax
        axis.clear()
        self._clean_axes(axis)
        history = result['history'] if result else []
        if history:
            generations = [row['generation'] for row in history]
            axis.plot(generations, [row['best'] for row in history], color=TEAL, linewidth=2.0, label='历史最优 F')
            axis.plot(generations, [row['mean'] for row in history], color=ORANGE, linewidth=1.5,
                      linestyle='--', label='种群平均 F')
            axis.legend(loc='lower right', frameon=False, fontsize=8, ncol=2)
            axis.set_xlabel('迭代代数', fontsize=8)
            axis.set_ylabel('综合得分 F', fontsize=8)
            axis.set_xlim(0, max(1, generations[-1]))
        else:
            text = ('选择遗传算法后，可观察推荐列表如何逐代优化' if not result else
                    '当前使用热门回退，无遗传迭代' if result['cold_start'] else
                    '暂无可推荐商品，未运行遗传迭代' if not result['returned'] else
                    '当前为确定性基线，无遗传迭代；切换遗传算法查看收敛过程')
            axis.text(0.5, 0.5, text, ha='center', va='center', color=MUTED, transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        self.convergence_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.94)
        self.convergence_canvas.draw_idle()
        for axis in self.profile_axes:
            axis.clear()
            self._clean_axes(axis)
        profile = result['profile'] if result else {}
        categories = list(profile)
        if categories:
            self.profile_axes[0].bar(categories, list(profile.values()), color=TEAL, width=0.6)
            self.profile_axes[0].set_ylim(0, 1.05)
            self.profile_axes[1].bar(categories, [result['candidate_categories'].get(cat, 0) for cat in categories],
                                     color=ORANGE, width=0.6)
        self.profile_axes[0].set_title('训练期用户画像', fontsize=9, loc='left', color=INK)
        self.profile_axes[1].set_title('候选商品类别分布 / 件', fontsize=9, loc='left', color=INK)
        self.profile_figure.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.80, wspace=0.25)
        self.profile_canvas.draw_idle()

    def export_to(self, path):
        if self.result is None:
            raise ValueError('请先生成推荐')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        self.status.set(f'已导出本次推荐：{path}')

    def export_dialog(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title='导出本次推荐', defaultextension='.json',
                                           initialdir=str(OUTPUT_ROOT),
                                           initialfile=f'recommendation_user{self.result["user_id"]}.json',
                                           filetypes=[('JSON 推荐结果', '*.json')])
        if path:
            try:
                self.export_to(path)
            except OSError as error:
                self.status.set('导出失败：' + str(error))

    def capture(self, path):
        """Capture this client window, excluding other windows and IME overlays."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        self.tree.focus_set()
        self.root.update()
        time.sleep(0.2)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os
            if os.name == 'nt':
                self._capture_windows_client(path)
            else:
                from PIL import ImageGrab
                x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
                ImageGrab.grab(bbox=(x, y, x + self.root.winfo_width(), y + self.root.winfo_height())).save(path)
        finally:
            self.root.attributes('-topmost', False)

    def _capture_windows_client(self, path):
        import ctypes
        from ctypes import wintypes
        from PIL import Image
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                   ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        hwnd = user32.GetAncestor(self.root.winfo_id(), 2)
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        width, height = rect.right, rect.bottom
        dc = user32.GetDC(hwnd)
        memory = gdi32.CreateCompatibleDC(dc)
        bitmap = gdi32.CreateCompatibleBitmap(dc, width, height)
        previous = gdi32.SelectObject(memory, bitmap)

        class BitmapInfo(ctypes.Structure):
            _fields_ = [('size', wintypes.DWORD), ('width', wintypes.LONG), ('height', wintypes.LONG),
                        ('planes', wintypes.WORD), ('bits', wintypes.WORD), ('compression', wintypes.DWORD),
                        ('image_size', wintypes.DWORD), ('x_resolution', wintypes.LONG),
                        ('y_resolution', wintypes.LONG), ('colors', wintypes.DWORD), ('important', wintypes.DWORD)]

        try:
            if not user32.PrintWindow(hwnd, memory, 3):
                raise OSError('窗口截图失败')
            info = BitmapInfo(ctypes.sizeof(BitmapInfo), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
            buffer = ctypes.create_string_buffer(width * height * 4)
            if not gdi32.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(info), 0):
                raise OSError('窗口像素读取失败')
            Image.frombuffer('RGB', (width, height), buffer, 'raw', 'BGRX', 0, 1).save(path)
        finally:
            gdi32.SelectObject(memory, previous)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memory)
            user32.ReleaseDC(hwnd, dc)

    def close(self):
        self.closed = True
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='本地商品推荐演示界面')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--smoke', action='store_true', help='执行真实界面冒烟检查并退出')
    parser.add_argument('--lightweight', action='store_true', help='使用原生轻量图表')
    parser.add_argument('--self-test', action='store_true', help='执行独立程序内置检查并退出')
    parser.add_argument('--output', type=Path, help='内置检查的JSON输出位置')
    args = parser.parse_args()
    if args.smoke:
        from tests.ui_smoke import run_smoke
        run_smoke(args.data)
        return
    enable_dpi_awareness()
    root = tk.Tk()
    if args.self_test:
        from recsys.portable_check import check
        root.attributes('-alpha', 0)
        app = RecommendationApp(root, args.data, auto_start=False)
        passed = check(app, args.output or OUTPUT_ROOT / 'portable_self_test.json')
        app.close()
        raise SystemExit(0 if passed else 1)
    RecommendationApp(root, args.data)
    root.mainloop()


if __name__ == '__main__':
    main()

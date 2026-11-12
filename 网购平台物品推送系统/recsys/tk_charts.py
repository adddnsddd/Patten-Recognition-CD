"""Small, dependency-free charts for the packaged Tk desktop application."""

import math
import tkinter as tk
from tkinter import font as tkfont


TEAL = '#087F8C'
ORANGE = '#E1974B'
MUTED = '#667D87'
INK = '#203743'
GRID = '#E8EEF0'
AXIS = '#C8D6DB'


class NativeCharts:
    """Draw recommendation results using only Tk Canvas and system fonts.

    ``draw(result=None)`` accepts the same result dictionary as the desktop
    application. Both public canvas attributes are ordinary ``tk.Canvas``
    widgets, so the caller can inspect or configure them when needed.
    """

    def __init__(self, convergence_frame, profile_frame):
        self.result = None
        self.font = tkfont.Font(root=convergence_frame, family='Microsoft YaHei UI', size=8)
        self.small_font = tkfont.Font(root=convergence_frame, family='Microsoft YaHei UI', size=7)
        self.title_font = tkfont.Font(root=convergence_frame, family='Microsoft YaHei UI',
                                      size=9, weight='bold')
        self.convergence_canvas = tk.Canvas(convergence_frame, background='white',
                                            height=175, highlightthickness=0, borderwidth=0)
        self.profile_canvas = tk.Canvas(profile_frame, background='white',
                                       height=175, highlightthickness=0, borderwidth=0)
        for canvas in (self.convergence_canvas, self.profile_canvas):
            canvas.pack(fill='both', expand=True)
        self.convergence_canvas.bind('<Configure>', self._convergence_changed, add='+')
        self.profile_canvas.bind('<Configure>', self._profile_changed, add='+')

    def draw(self, result=None):
        """Remember a result and redraw both charts at their current sizes."""
        self.result = result
        self._draw_convergence()
        self._draw_profile()

    def _convergence_changed(self, _event):
        self._draw_convergence()

    def _profile_changed(self, _event):
        self._draw_profile()

    @staticmethod
    def _size(canvas):
        # Unmapped notebook pages initially report one pixel. Their Configure
        # event will draw them once they have a usable size.
        return canvas.winfo_width(), canvas.winfo_height()

    @staticmethod
    def _number(value, default=0.0):
        try:
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _nice_step(span, intervals, integer=False):
        raw = max(span, 1e-9) / max(1, intervals)
        power = 10 ** math.floor(math.log10(raw))
        fraction = raw / power
        factor = next((value for value in (1, 2, 2.5, 5, 10) if value >= fraction), 10)
        step = factor * power
        return max(1, math.ceil(step)) if integer else step

    @staticmethod
    def _format(value):
        return f'{value:.3f}'.rstrip('0').rstrip('.') or '0'

    def _text(self, canvas, x, y, text, **options):
        options.setdefault('font', self.font)
        options.setdefault('fill', MUTED)
        return canvas.create_text(x, y, text=text, **options)

    def _message(self, canvas, bounds, text):
        left, top, right, bottom = bounds
        self._text(canvas, (left + right) / 2, (top + bottom) / 2, text,
                   width=max(30, right - left - 24), justify='center', tags='message')

    def _draw_convergence(self):
        canvas = self.convergence_canvas
        canvas.delete('all')
        width, height = self._size(canvas)
        if width < 80 or height < 55:
            return
        result = self.result
        history = (result or {}).get('history') or []
        if not history:
            if result is None:
                text = '选择遗传算法，查看推荐列表的逐代优化'
            elif not result.get('returned', 0):
                text = '暂无可推荐商品，未运行遗传迭代'
            elif result.get('cold_start', False):
                text = '当前使用热门回退，无遗传迭代'
            else:
                text = '当前为确定性基线，无遗传迭代\n切换遗传算法查看收敛过程'
            self._message(canvas, (0, 0, width, height), text)
            return

        line_height = self.font.metrics('linespace')
        top = line_height + 18
        bottom = height - 2 * line_height - 13
        left = max(49, self.font.measure('0.000') + 17)
        right = width - 18
        if bottom <= top + 12 or right <= left + 12:
            return
        self._text(canvas, 12, 7, '综合得分 F', anchor='nw', fill=INK)
        legend_width = self.font.measure('历史最优 F') + self.font.measure('种群平均 F') + 86
        legend_x = max(left, right - legend_width)
        if width >= legend_width + self.font.measure('综合得分 F') + 40:
            for label, color, dashed in [('历史最优 F', TEAL, False), ('种群平均 F', ORANGE, True)]:
                canvas.create_line(legend_x, 8 + line_height / 2, legend_x + 20,
                                   8 + line_height / 2, fill=color, width=2,
                                   dash=(5, 3) if dashed else (), tags='legend')
                self._text(canvas, legend_x + 26, 8, label, anchor='nw', tags='legend')
                legend_x += self.font.measure(label) + 44
        else:
            # At a narrow width, keep the complete legend on its own line.
            self._text(canvas, right, 7, '青：最优  橙虚线：平均', anchor='ne',
                       font=self.small_font, tags='legend')

        points = [(self._number(row.get('generation', index)),
                   self._number(row.get('best')), self._number(row.get('mean')))
                  for index, row in enumerate(history)]
        points.sort(key=lambda row: row[0])
        x_min = min(0, points[0][0])
        x_max = max(x_min + 1, points[-1][0])
        values = [value for row in points for value in row[1:]]
        data_min, data_max = min(values), max(values)
        padding = max(0.015, (data_max - data_min) * 0.12)
        y_step = self._nice_step(max(data_max - data_min + padding * 2, 0.06),
                                 4 if bottom - top >= 100 else 3)
        y_min = math.floor((data_min - padding) / y_step) * y_step
        y_max = math.ceil((data_max + padding) / y_step) * y_step
        if data_min >= 0:
            y_min = max(0, y_min)
        if data_max <= 1:
            y_max = min(1, y_max)
        y_max = max(y_min + 1e-9, y_max)

        def x_position(value):
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def y_position(value):
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        y_ticks = [y_min]
        next_tick = (math.floor(y_min / y_step) + 1) * y_step
        while next_tick < y_max - y_step * 1e-5:
            y_ticks.append(next_tick)
            next_tick += y_step
        y_ticks.append(y_max)
        for value in y_ticks:
            y = y_position(value)
            canvas.create_line(left, y, right, y, fill=GRID, tags='grid')
            self._text(canvas, left - 8, y, self._format(value), anchor='e', tags='tick')
        x_step = self._nice_step(x_max - x_min, max(1, min(7, int((right - left) / 95))), True)
        x_ticks = [x_min]
        next_tick = (math.floor(x_min / x_step) + 1) * x_step
        while next_tick < x_max:
            x_ticks.append(next_tick)
            next_tick += x_step
        if len(x_ticks) > 1 and (x_max - x_ticks[-1]) / (x_max - x_min) * (right - left) < 38:
            x_ticks.pop()
        x_ticks.append(x_max)
        for value in x_ticks:
            x = x_position(value)
            canvas.create_line(x, top, x, bottom, fill=GRID, tags='grid')
            self._text(canvas, x, bottom + 5, self._format(value), anchor='n', tags='tick')
        canvas.create_line(left, top, left, bottom, right, bottom, fill=AXIS, tags='axis')
        self._text(canvas, (left + right) / 2, height - 4, '迭代代数', anchor='s')
        for column, color, dashed in [(1, TEAL, False), (2, ORANGE, True)]:
            coordinates = [coordinate for row in points
                           for coordinate in (x_position(row[0]), y_position(row[column]))]
            if len(points) > 1:
                canvas.create_line(*coordinates, fill=color, width=2,
                                   dash=(6, 4) if dashed else (), tags='series')
            else:
                x, y = coordinates
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                   fill=color, outline='white', tags='series')

    def _wrap_label(self, text, width, font):
        """Wrap complete category names instead of truncating them."""
        lines, current = [], ''
        for character in str(text):
            if character == '\n' or (current and font.measure(current + character) > width):
                lines.append(current)
                current = '' if character == '\n' else character
            else:
                current += character
        lines.append(current)
        return '\n'.join(lines)

    def _draw_profile(self):
        canvas = self.profile_canvas
        canvas.delete('all')
        width, height = self._size(canvas)
        if width < 120 or height < 65:
            return
        result = self.result or {}
        profile = result.get('profile') or {}
        candidates = result.get('candidate_categories') or {}
        categories = list(profile)
        categories.extend(category for category in candidates if category not in profile)
        gap = 22 if width >= 650 else 10
        half = (width - gap) / 2
        self._draw_bars(canvas, (0, 0, half, height), categories,
                        [max(0, min(1, self._number(profile.get(cat)))) for cat in categories],
                        '训练期用户画像', TEAL, False)
        self._draw_bars(canvas, (half + gap, 0, width, height), categories,
                        [max(0, int(self._number(candidates.get(cat)))) for cat in categories],
                        '候选商品类别分布 / 件', ORANGE, True)

    def _draw_bars(self, canvas, bounds, categories, values, title, color, integer):
        panel_left, _, panel_right, height = bounds
        panel_width = panel_right - panel_left
        title_font = self.title_font if panel_width >= 260 else self.font
        self._text(canvas, panel_left + 12, 7, title, anchor='nw', font=title_font, fill=INK,
                   width=max(30, panel_width - 20), tags='title')
        title_height = title_font.metrics('linespace')
        if not categories:
            if self.result is None:
                message = '生成推荐后查看候选分布' if integer else '生成推荐后查看用户画像'
            elif integer:
                message = '暂无候选商品'
            elif self.result.get('cold_start'):
                message = '暂无历史偏好（冷启动）'
            else:
                message = '暂无画像数据'
            self._message(canvas, (panel_left, title_height + 14, panel_right, height), message)
            return

        highest = max(values, default=0)
        if integer:
            step = self._nice_step(max(1, highest) * 1.18, 3, True)
            y_max = max(step, math.ceil(max(1, highest) * 1.18 / step) * step)
        else:
            y_max = 1.0
            step = 0.5 if height < 190 else 0.25
        y_label_width = max(self.font.measure(self._format(y_max)), self.font.measure('0.5'))
        left = panel_left + y_label_width + 18
        right = panel_right - 10
        cell_width = max(1, (right - left) / len(categories))
        label_font = self.font if cell_width >= 29 else self.small_font
        labels = [self._wrap_label(category, max(10, cell_width - 3), label_font)
                  for category in categories]
        label_height = max(label.count('\n') + 1 for label in labels) * label_font.metrics('linespace')
        top = title_height + self.font.metrics('linespace') + 16
        bottom = height - label_height - 10
        if bottom < top + 12:
            top = title_height + 16
        if bottom <= top:
            return
        ticks = [index * step for index in range(int(round(y_max / step)) + 1)]
        for value in ticks:
            y = bottom - value / y_max * (bottom - top)
            canvas.create_line(left, y, right, y, fill=GRID, tags='grid')
            self._text(canvas, left - 6, y, self._format(value), anchor='e', tags='tick')
        canvas.create_line(left, top, left, bottom, right, bottom, fill=AXIS, tags='axis')
        for index, (label, value) in enumerate(zip(labels, values)):
            x = left + (index + 0.5) * cell_width
            y = bottom - value / y_max * (bottom - top)
            bar_width = max(1, min(36, cell_width * 0.58))
            canvas.create_rectangle(x - bar_width / 2, y, x + bar_width / 2, bottom,
                                    fill=color, outline='', tags='bar')
            self._text(canvas, x, max(title_height + 11, y - 3),
                       str(int(value)) if integer else f'{value:.2f}'.rstrip('0').rstrip('.'),
                       anchor='s', font=label_font, fill=color, tags='value')
            self._text(canvas, x, bottom + 5, label, anchor='n', justify='center',
                       font=label_font, fill=INK, tags='category')
        if not integer and not any(values) and self.result and self.result.get('cold_start'):
            self._message(canvas, (left, top, right, bottom - self.font.metrics('linespace')),
                          '暂无历史偏好（冷启动）')

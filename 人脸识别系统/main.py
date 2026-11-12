"""Portable entry point for the entirely local face-recognition application."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faceid import runtime


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # A windowed EXE has no console; main persists this as JSON and a log.
        raise ValueError("命令行参数错误：" + message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="本地人脸识别系统：PCA 与 LBPH")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--self-test", action="store_true", help="源码运行完整功能测试；独立程序运行便携检查；不访问摄像头")
    action.add_argument("--portable-self-test", action="store_true", help="运行独立程序便携检查；不访问摄像头")
    action.add_argument("--ui-smoke", action="store_true", help="创建并关闭桌面窗口；不访问摄像头")
    action.add_argument("--benchmark-video", type=Path, help="对指定本地视频运行测速，末尾循环")
    action.add_argument("--stability-video", type=Path, help="对指定本地视频运行稳定性检查，末尾循环")
    action.add_argument("--audit-data", action="store_true", help="检查本地数据清单")
    action.add_argument("--environment", action="store_true", help="记录本机软硬件环境")
    action.add_argument("--evaluate-detection", nargs="?", const=True, type=Path, help="使用标注CSV评价检测召回率；省略路径时用默认标注")
    parser.add_argument("--verify-files", action="store_true", help="数据检查时校验原始图像文件")
    parser.add_argument("--method", choices=["pca", "lbph"], default="lbph", help="性能检查方法")
    parser.add_argument("--frames", type=int, default=300, help="每次测速帧数；非300仅属调试")
    parser.add_argument("--repeats", type=int, default=3, help="测速重复次数；非3仅属调试")
    parser.add_argument("--duration", type=float, default=600, help="稳定性秒数；小于600仅属调试")
    parser.add_argument("--allow-untrained", action="store_true", help="明确允许仅测通路；不算正式验收")
    parser.add_argument("--data-dir", type=Path, help="指定数据目录；相对路径按当前工作目录解释")
    parser.add_argument("--result-file", type=Path, help="将本次检查结果保存为UTF-8 JSON，适用于无控制台独立程序")
    return parser


def _result_file_argument(arguments: list[str]) -> Path | None:
    """Recover the output path even when other arguments are invalid."""
    for index, value in enumerate(arguments):
        if value.startswith("--result-file="):
            return Path(value.partition("=")[2])
        if value == "--result-file" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
    return None


def _attach_runtime(result: dict, data_root: Path, explicit: bool) -> dict:
    if not isinstance(result, dict):
        raise TypeError("检查程序未返回有效的JSON结果。")
    result = dict(result)
    result["runtime"] = runtime.runtime_info(data_root, explicit=explicit)
    result.setdefault("data_root", str(data_root))
    return result


def _run_gui(args, data_root: Path) -> dict | None:
    import tkinter as tk

    from faceid.gui import FaceApp, enable_windows_dpi_awareness
    from faceid.service import RecognitionService

    enable_windows_dpi_awareness()
    root = tk.Tk()
    if args.ui_smoke:
        root.withdraw()
    try:
        app = FaceApp(root, RecognitionService(data_root))
        if args.ui_smoke:
            root.update_idletasks()
            root.update()
            assert not app._camera_running, "UI smoke must not activate the camera"
            assert app._camera_thread is None
            assert app.train_button.winfo_exists()
            assert app.capture_button.winfo_exists()
            assert app.import_button.winfo_exists()
            assert app.evaluate_button.winfo_exists()
            assert app.open_video_button.winfo_exists()
            assert app.benchmark_button.winfo_exists()
            assert app.stability_button.winfo_exists()
            assert len(app.notebook.tabs()) == 4
            app.close()
            return dict(ui_smoke="passed", passed=True, camera_accessed=False, tabs=4)
        root.mainloop()
        return None
    except BaseException:
        try:
            root.destroy()
        except tk.TclError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    result_file = _result_file_argument(arguments)
    data_root = None
    args = None
    normal_gui = False
    try:
        args = _parser().parse_args(arguments)
        result_file = args.result_file
        has_cli_action = bool(args.self_test or args.portable_self_test or args.ui_smoke or args.audit_data
                              or args.environment or args.evaluate_detection is not None
                              or args.benchmark_video or args.stability_video)
        normal_gui = not has_cli_action and result_file is None
        data_root = runtime.resolve_data_root(args.data_dir)
        runtime.ensure_writable_data_root(data_root)
        if args.portable_self_test or (args.self_test and runtime.is_frozen()):
            from faceid.portable_check import run_portable_check

            result = run_portable_check(data_root)
            code = 0 if result.get("passed") is True else 1
        elif args.self_test:
            from faceid.selftest import run_self_test

            result = run_self_test()
            code = 0 if isinstance(result, dict) and result.get("passed") is True else 1
        elif args.audit_data or args.environment or args.evaluate_detection is not None or args.benchmark_video or args.stability_video:
            from faceid.service import RecognitionService

            service = RecognitionService(data_root)
            if args.audit_data:
                result = service.audit_data(verify_files=args.verify_files)
                code = 0 if result.get("passed") else 1
            elif args.environment:
                result = service.environment()
                code = 0
            elif args.evaluate_detection is not None:
                result = service.evaluate_detection(None if args.evaluate_detection is True else args.evaluate_detection)
                code = 0
            else:
                from faceid.benchmark import run_benchmark, run_stability

                try:
                    service.load_models()
                except (ValueError, FileNotFoundError):
                    if not args.allow_untrained:
                        raise
                if args.benchmark_video:
                    result = run_benchmark(service, args.benchmark_video, method=args.method,
                                           frames=args.frames, repeats=args.repeats, allow_untrained=args.allow_untrained)
                else:
                    result = run_stability(service, args.stability_video, method=args.method,
                                           duration_seconds=args.duration, allow_untrained=args.allow_untrained)
                code = 0 if result.get("completed") else 1
        else:
            result = _run_gui(args, data_root)
            if result is None:
                return 0
            code = 0
        runtime.emit_json(_attach_runtime(result, data_root, args.data_dir is not None), result_file)
        return code
    except Exception as exc:
        log_path = runtime.log_startup_error(exc, data_root)
        message = "启动或检查失败：%s" % exc
        if isinstance(exc, ImportError):
            message += "\n" + ("独立程序组件缺失，请重新解压完整程序。" if runtime.is_frozen()
                                 else "请在项目目录执行：python -m pip install -r requirements.txt")
        if isinstance(exc, PermissionError):
            message += "\n请通过 --data-dir 指定可写的数据目录。"
        message += "\n错误日志：" + (str(log_path) if log_path is not None else "无法写入错误日志，请检查目录权限。")
        result = dict(passed=False, error=message, exception=type(exc).__name__,
                      error_log=str(log_path) if log_path is not None else None)
        if data_root is not None:
            result = _attach_runtime(result, data_root, args is not None and args.data_dir is not None)
        try:
            runtime.emit_json(result, result_file, error=True)
        except Exception as output_error:
            runtime.safe_write(message + "\n结果JSON无法写入：%s\n" % output_error, error=True)
        if normal_gui:
            runtime.show_startup_error(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

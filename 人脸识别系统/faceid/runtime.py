"""Portable application paths and diagnostics, independent of the caller cwd.

Resources may live in PyInstaller's extraction folder. User data must never be
stored there: frozen applications keep their own named data folder by the EXE.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path


DATA_FOLDER_NAME = "人脸识别数据"


class DataDirectoryError(OSError):
    """The requested data location could not be used; no fallback is made."""


def absolute_path(value: str | Path) -> Path:
    """Resolve only after making the path absolute, including missing folders.

    Some older Windows Python builds return a relative path from resolve() for
    a nonexistent relative target. abspath() preserves the caller-cwd contract.
    """
    return Path(os.path.abspath(str(Path(value).expanduser()))).resolve()


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """The source project folder or the directory containing the frozen EXE."""
    if is_frozen():
        return absolute_path(sys.executable).parent
    return absolute_path(__file__).parent.parent


def resource_root() -> Path:
    """Read-only bundled resources, not a location for samples or models."""
    if is_frozen() and getattr(sys, "_MEIPASS", None):
        return absolute_path(sys._MEIPASS)
    return app_root()


def default_data_root() -> Path:
    return app_root() / DATA_FOLDER_NAME if is_frozen() else app_root()


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Explicit relative paths follow the caller cwd, never the package folder."""
    if explicit is not None:
        return absolute_path(explicit)
    return absolute_path(default_data_root())


def ensure_writable_data_root(root: str | Path) -> Path:
    """Check the exact requested folders without silently choosing another."""
    target = absolute_path(root)
    try:
        for folder in (target, target / "data", target / "models", target / "outputs"):
            folder.mkdir(parents=True, exist_ok=True)
            # A real write is more reliable than os.access on Windows ACLs.
            with tempfile.TemporaryFile(prefix=".faceid-write-", dir=folder) as stream:
                stream.write(b"faceid")
                stream.flush()
    except OSError as exc:
        raise DataDirectoryError(
            "无法写入人脸识别数据目录：%s\n请将程序放到可写文件夹，或通过 --data-dir 指定可写目录；数据未切换到其他位置。\n原始错误：%s"
            % (target, exc)
        ) from exc
    return target


def runtime_info(data_root: str | Path, explicit: bool = False) -> dict:
    return dict(frozen=is_frozen(), executable=str(absolute_path(sys.executable)),
                app_root=str(app_root()), resource_root=str(resource_root()),
                data_root=str(absolute_path(data_root)), cwd=str(Path.cwd()),
                data_root_from="explicit" if explicit else "executable_directory" if is_frozen() else "source_directory")


def safe_write(text: str, *, error: bool = False) -> None:
    """PyInstaller --windowed sets both streams to None; that is normal."""
    stream = getattr(sys, "stderr" if error else "stdout", None)
    if stream is None:
        return
    try:
        stream.write(text)
        stream.flush()
    except UnicodeError:
        try:
            encoding = getattr(stream, "encoding", None) or "utf-8"
            stream.write(text.encode(encoding, errors="backslashreplace").decode(encoding))
            stream.flush()
        except (OSError, UnicodeError, ValueError, AttributeError):
            pass
    except (OSError, ValueError, AttributeError):
        pass


def emit_json(result: dict, result_file: str | Path | None = None, *, error: bool = False) -> Path | None:
    """Atomically publish UTF-8 JSON, then optionally write a safe console copy."""
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"
    destination = absolute_path(result_file) if result_file is not None else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False,
                                             dir=destination.parent, prefix=".faceid-result-", suffix=".tmp") as stream:
                temporary = Path(stream.name)
                stream.write(text)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    safe_write(text, error=error)
    return destination


def log_startup_error(exc: BaseException, data_root: str | Path | None = None) -> Path | None:
    """Persist diagnostics even when the requested data directory is read-only.

    Only the diagnostic log may fall back. The application never changes its
    sample/model location, and callers show the actual log path to the user.
    """
    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root) / "outputs" / "startup_error.log")
    candidates.append(app_root() / "startup_error.log")
    candidates.append(Path(tempfile.gettempdir()) / ("faceid-startup-error-%d.log" % os.getpid()))
    detail = "[%s]\n%s\n" % (datetime.now().isoformat(timespec="seconds"), "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)))
    for destination in candidates:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(detail)
            return destination.resolve()
        except OSError:
            continue
    return None


def show_startup_error(message: str) -> None:
    """Show a visible error only for normal interactive GUI startup."""
    try:
        from tkinter import messagebox
        messagebox.showerror("人脸识别系统启动失败", message)
    except Exception:
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, message, "人脸识别系统启动失败", 0x10)
            except Exception:
                pass

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import CaptureConfig
from .domain import CaptureResult, GameWindow, Rect
from .identity import capture_pair_id_from_path, session_id_from_pair_id

try:
    import mss
    from PIL import Image
except Exception:  # pragma: no cover
    mss = None
    Image = None

try:
    import win32con
    import win32gui
except Exception:  # pragma: no cover
    win32con = None
    win32gui = None


class CaptureError(RuntimeError):
    pass


def find_game_window(title_keyword: str) -> GameWindow:
    if win32gui is None:
        raise CaptureError("pywin32 is required to find the game window on Windows.")

    matches: list[GameWindow] = []

    def callback(hwnd: int, _extra: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_keyword.lower() not in title.lower():
            return
        client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (client_left, client_top))
        screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (client_right, client_bottom))
        matches.append(
            GameWindow(
                hwnd=hwnd,
                title=title,
                client_rect=Rect(screen_left, screen_top, screen_right, screen_bottom),
            )
        )

    win32gui.EnumWindows(callback, None)
    if not matches:
        raise CaptureError(f"Game window not found: {title_keyword}")
    return matches[0]


def is_point_in_client(window: GameWindow, x: int, y: int) -> bool:
    return window.client_rect.contains(x, y)


def build_capture_rect(window: GameWindow, mouse_x: int, mouse_y: int, config: CaptureConfig) -> Rect:
    desired = Rect(
        mouse_x - config.left,
        mouse_y - config.up,
        mouse_x + config.right,
        mouse_y + config.down,
    )
    return desired.clamp_within(window.client_rect)


def capture_rect_to_file(rect: Rect, output_dir: Path, prefix: str = "capture") -> Path:
    if mss is None or Image is None:
        raise CaptureError("mss and pillow are required for screen capture.")
    if rect.width <= 0 or rect.height <= 0:
        raise CaptureError("Invalid capture rectangle.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.png")

    with mss.mss() as sct:
        monitor = {"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height}
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        image.save(output_path)

    return output_path


def capture_mouse_region(window: GameWindow, mouse_x: int, mouse_y: int, config: CaptureConfig) -> CaptureResult:
    if not is_point_in_client(window, mouse_x, mouse_y):
        raise CaptureError("Mouse is outside the game client area.")

    rect = build_capture_rect(window, mouse_x, mouse_y, config)
    path = capture_rect_to_file(rect, config.output_dir)
    pair_id = capture_pair_id_from_path(path)
    return CaptureResult(
        image_path=path,
        capture_rect=rect,
        mouse_x=mouse_x,
        mouse_y=mouse_y,
        captured_at=datetime.now(),
        capture_pair_id=pair_id,
        session_id=session_id_from_pair_id(pair_id),
    )


def capture_game_client(window: GameWindow, config: CaptureConfig, prefix: str) -> CaptureResult:
    rect = window.client_rect
    path = capture_rect_to_file(rect, config.output_dir, prefix=prefix)
    pair_id = capture_pair_id_from_path(path)
    return CaptureResult(
        image_path=path,
        capture_rect=rect,
        mouse_x=0,
        mouse_y=0,
        captured_at=datetime.now(),
        capture_pair_id=pair_id,
        session_id=session_id_from_pair_id(pair_id),
    )


def bring_window_to_front(window: GameWindow) -> None:
    if win32gui is None or win32con is None:
        return
    win32gui.ShowWindow(window.hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetForegroundWindow(window.hwnd)

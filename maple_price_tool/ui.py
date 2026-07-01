from __future__ import annotations

import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .capture import CaptureError, capture_game_client, find_game_window
from .config import AppConfig, save_config
from .domain import AnalysisResult, AppState, CaptureResult, Rect
from .identity import capture_pair_id_from_path, parse_sidecar, session_id_from_pair_id, sidecar_payload
from .storage import Storage, final_record_from_analysis
from .vision import OpenCvTemplateRecognizer
from recognition.training_samples import SampleSaveSummary, TrainingSampleWriter

try:
    from pynput import keyboard
except Exception:  # pragma: no cover
    keyboard = None


logger = logging.getLogger(__name__)
PREVIEW_SIZE = QSize(520, 390)
CROP_THUMB_MAX = QSize(150, 40)
PRICE_CROP_THUMB_MAX = QSize(260, 72)


@dataclass
class CaptureJob:
    config: AppConfig
    prefix: str
    before_image_path: Path | None = None


@dataclass
class AnalyzeImageJob:
    config: AppConfig
    image_path: Path


class CaptureOnlyWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job: CaptureJob) -> None:
        super().__init__()
        self.job = job

    @Slot()
    def run(self) -> None:
        try:
            window = find_game_window(self.job.config.window_title_keyword)
            capture = capture_game_client(window, self.job.config.capture, prefix=self.job.prefix)
            if self.job.before_image_path is not None:
                capture_pair_id = capture_pair_id_from_path(self.job.before_image_path)
                capture = CaptureResult(
                    image_path=capture.image_path,
                    capture_rect=capture.capture_rect,
                    mouse_x=capture.mouse_x,
                    mouse_y=capture.mouse_y,
                    captured_at=capture.captured_at,
                    before_image_path=self.job.before_image_path,
                    capture_pair_id=capture_pair_id,
                    session_id=session_id_from_pair_id(capture_pair_id),
                )
            logger.info("capture saved image=%s", capture.image_path)
            self.finished.emit(capture)
        except Exception as exc:
            logger.exception("capture failed")
            self.failed.emit(str(exc))


class AnalyzeImageWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job: AnalyzeImageJob) -> None:
        super().__init__()
        self.job = job

    @Slot()
    def run(self) -> None:
        try:
            image_path = self.job.image_path
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            capture = CaptureResult(
                image_path=image_path,
                capture_rect=Rect(0, 0, 0, 0),
                mouse_x=0,
                mouse_y=0,
                captured_at=datetime.fromtimestamp(image_path.stat().st_mtime),
                before_image_path=read_before_sidecar(image_path),
                capture_pair_id=read_capture_pair_id_sidecar(image_path) or capture_pair_id_from_path(image_path),
                session_id=read_session_id_sidecar(image_path) or session_id_from_pair_id(
                    read_capture_pair_id_sidecar(image_path) or capture_pair_id_from_path(image_path)
                ),
            )
            recognizer = OpenCvTemplateRecognizer(self.job.config.vision, self.job.config.capture.debug_dir)
            analysis = recognizer.analyze(capture)
            logger.info("analysis recognizer=%s image=%s", recognizer.name, image_path)
            self.finished.emit(analysis)
        except Exception as exc:
            logger.exception("analysis failed")
            self.failed.emit(str(exc))


class ScaledImageLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(PREVIEW_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #777; }")

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()

    def clear_source(self, text: str = "") -> None:
        self._source_pixmap = QPixmap()
        self.clear()
        self.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ReviewWindow(QMainWindow):
    def __init__(self, config: AppConfig, storage: Storage) -> None:
        super().__init__()
        self.config = config
        self.storage = storage
        self.state = AppState.IDLE
        self.analysis: AnalysisResult | None = None
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None
        self.hotkey_listener = None
        self.latest_before_path: Path | None = None

        self.setWindowTitle("Maple Auction Review MVP")
        self.setMinimumSize(1180, 840)
        self._build_menu()
        self._build_ui()
        self.refresh_capture_files()
        self.latest_before_path = self.find_latest_before_capture()
        self._start_hotkey()

    def _build_menu(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")
        capture_action = settings_menu.addAction("Capture Region...")
        capture_action.triggered.connect(self.open_capture_settings)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.status_label = QLabel("IDLE - F7 captures before, F8 captures after using latest F7.")
        layout.addWidget(self.status_label)

        file_row = QHBoxLayout()
        self.capture_selector = QComboBox()
        self.capture_selector.setMinimumWidth(360)
        self.capture_selector.currentIndexChanged.connect(self.update_image_preview)
        self.previous_button = QPushButton("←")
        self.next_button = QPushButton("→")
        self.refresh_button = QPushButton("Refresh")
        self.analyze_button = QPushButton("Analyze Selected PNG")
        self.previous_button.setToolTip("Previous image")
        self.next_button.setToolTip("Next image")
        self.previous_button.clicked.connect(self.select_previous_capture)
        self.next_button.clicked.connect(self.select_next_capture)
        self.refresh_button.clicked.connect(self.refresh_capture_files)
        self.analyze_button.clicked.connect(self.start_analysis_selected)
        file_row.addWidget(QLabel("Capture PNG"))
        file_row.addWidget(self.capture_selector, 1)
        file_row.addWidget(self.previous_button)
        file_row.addWidget(self.next_button)
        file_row.addWidget(self.refresh_button)
        file_row.addWidget(self.analyze_button)
        layout.addLayout(file_row)

        self.preview_label = ScaledImageLabel("No image selected")
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumSize(PREVIEW_SIZE.width() + 18, PREVIEW_SIZE.height() + 18)
        self.preview_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_scroll.setWidget(self.preview_label)

        self.diff_preview_label = ScaledImageLabel("Analyze 후 diff 표시")
        self.diff_preview_scroll = QScrollArea()
        self.diff_preview_scroll.setWidgetResizable(True)
        self.diff_preview_scroll.setMinimumSize(PREVIEW_SIZE.width() + 18, PREVIEW_SIZE.height() + 18)
        self.diff_preview_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.diff_preview_scroll.setWidget(self.diff_preview_label)

        self.crop_list_widget = QWidget()
        self.crop_rows_layout = QVBoxLayout(self.crop_list_widget)
        self.crop_rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.crop_preview_scroll = QScrollArea()
        self.crop_preview_scroll.setWidgetResizable(True)
        self.crop_preview_scroll.setMinimumWidth(540)
        self.crop_preview_scroll.setWidget(self.crop_list_widget)
        self.clear_crop_preview("Analyze 후 crop 표시")

        form = QFormLayout()
        self.fields: dict[str, QLineEdit | QTextEdit] = {
            "req_level": QLineEdit(),
            "equipment_type": QLineEdit(),
            "price_meso": QLineEdit(),
            "equipment_options": QTextEdit(),
            "potential": QTextEdit(),
        }
        form.addRow("REQ LEV", self.fields["req_level"])
        form.addRow("장비분류", self.fields["equipment_type"])
        form.addRow("가격(메소)", self.fields["price_meso"])
        form.addRow("장비옵션", self.fields["equipment_options"])
        form.addRow("잠재능력", self.fields["potential"])

        self.confidence_label = QLabel("confidence: -")
        self.training_label = QLabel("training samples: -")
        self.image_label = QLabel("image: -")

        self.save_button = QPushButton("Save (Enter / Ctrl+S)")
        self.cancel_button = QPushButton("Cancel (Esc)")
        self.save_button.clicked.connect(self.save_current)
        self.cancel_button.clicked.connect(self.cancel_review)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        image_row = QHBoxLayout()
        before_column = QVBoxLayout()
        before_column.addWidget(QLabel("Selected PNG"))
        before_column.addWidget(self.preview_scroll)
        residual_column = QVBoxLayout()
        residual_column.addWidget(QLabel("Residual"))
        residual_column.addWidget(self.diff_preview_scroll)
        image_row.addLayout(before_column, 1)
        image_row.addLayout(residual_column, 1)
        left_layout.addLayout(image_row, 1)
        left_layout.addLayout(form)
        left_layout.addWidget(self.confidence_label)
        left_layout.addWidget(self.training_label)
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.save_button)
        left_layout.addWidget(self.cancel_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Crop Preview"))
        right_layout.addWidget(self.crop_preview_scroll, 1)

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([1, 1])
        layout.addWidget(main_splitter, 1)

        self.setCentralWidget(central)

    def _start_hotkey(self) -> None:
        if keyboard is None:
            self.set_status(AppState.ERROR, "pynput is not installed; F7/F8 disabled")
            return

        def on_press(key) -> None:
            if key == keyboard.Key.f7:
                QApplication.postEvent(self, _F7Event())
            if key == keyboard.Key.f8:
                QApplication.postEvent(self, _F8Event())

        self.hotkey_listener = keyboard.Listener(on_press=on_press)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()

    def customEvent(self, event) -> None:  # noqa: N802 - Qt API
        if isinstance(event, _F7Event):
            self.start_before_capture()
            return
        if isinstance(event, _F8Event):
            self.start_after_capture()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() in (16777220, 16777221):
            self.save_current()
            return
        if event.key() == 16777216:
            self.cancel_review()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_S:
            self.save_current()
            return
        super().keyPressEvent(event)

    def set_status(self, state: AppState, message: str) -> None:
        self.state = state
        self.status_label.setText(f"{state.value} - {message}")
        logger.info("%s: %s", state.value, message)

    def refresh_capture_files(self, selected: Path | None = None) -> None:
        capture_dir = self.config.capture.output_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(capture_dir.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
        selected_text = str(selected) if selected is not None else self.capture_selector.currentData()

        self.capture_selector.blockSignals(True)
        self.capture_selector.clear()
        for path in files:
            label = f"{path.name}  ({datetime.fromtimestamp(path.stat().st_mtime).strftime('%H:%M:%S')})"
            self.capture_selector.addItem(label, str(path))
        if selected_text:
            index = self.capture_selector.findData(str(selected_text))
            if index >= 0:
                self.capture_selector.setCurrentIndex(index)
        self.capture_selector.blockSignals(False)
        self.update_image_preview()

    def find_latest_before_capture(self) -> Path | None:
        capture_dir = self.config.capture.output_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(capture_dir.glob("before_*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def selected_capture_path(self) -> Path | None:
        value = self.capture_selector.currentData()
        if not value:
            return None
        return Path(str(value))

    def select_previous_capture(self) -> None:
        count = self.capture_selector.count()
        if count <= 1:
            return
        index = max(0, self.capture_selector.currentIndex() - 1)
        self.capture_selector.setCurrentIndex(index)

    def select_next_capture(self) -> None:
        count = self.capture_selector.count()
        if count <= 1:
            return
        index = min(count - 1, self.capture_selector.currentIndex() + 1)
        self.capture_selector.setCurrentIndex(index)

    def update_image_preview(self) -> None:
        image_path = self.selected_capture_path()
        if image_path is None:
            self.preview_label.clear_source("No image selected")
            self.preview_label.setMinimumSize(PREVIEW_SIZE)
            self.clear_diff_preview("Analyze 후 diff 표시")
            self.clear_crop_preview("Analyze 후 crop 표시")
            return
        self.show_before_preview_for_capture(image_path)
        self.show_diff_preview_for_capture(image_path)
        self.clear_crop_preview("Analyze 후 crop 표시")

    def start_before_capture(self) -> None:
        if self.state in {AppState.CAPTURING, AppState.ANALYZING, AppState.SAVING}:
            logger.info("ignored F7 while busy: %s", self.state.value)
            return
        self.set_status(AppState.CAPTURING, "capturing full game screen as before")
        self._run_worker(
            CaptureOnlyWorker(CaptureJob(self.config, prefix="before")),
            finished=self.on_capture_ready,
            failed=self.on_capture_failed,
        )

    def start_after_capture(self) -> None:
        if self.state in {AppState.CAPTURING, AppState.ANALYZING, AppState.SAVING}:
            logger.info("ignored F8 while busy: %s", self.state.value)
            return
        before_path = self.latest_before_path if self.latest_before_path and self.latest_before_path.exists() else None
        before_path = before_path or self.find_latest_before_capture()
        if before_path is None:
            self.set_status(AppState.IDLE, "press F7 first to capture before image")
            QMessageBox.information(self, "Before capture required", "Press F7 first to capture the game screen before hover.")
            return
        self.latest_before_path = before_path
        self.set_status(AppState.CAPTURING, f"capturing full game screen as after using {before_path.name}")
        self._run_worker(
            CaptureOnlyWorker(CaptureJob(self.config, prefix="after", before_image_path=before_path)),
            finished=self.on_capture_ready,
            failed=self.on_capture_failed,
        )

    def start_analysis_selected(self) -> None:
        if self.state in {AppState.CAPTURING, AppState.ANALYZING, AppState.SAVING}:
            logger.info("ignored analyze while busy: %s", self.state.value)
            return
        image_path = self.selected_capture_path()
        if image_path is None:
            QMessageBox.information(self, "No capture selected", "Select a PNG file first.")
            return
        self.set_status(AppState.ANALYZING, f"analyzing {image_path.name}")
        self.clear_crop_preview("Analyzing...")
        self._run_worker(
            AnalyzeImageWorker(AnalyzeImageJob(self.config, image_path)),
            finished=self.on_analysis_ready,
            failed=self.on_analysis_failed,
        )

    def _run_worker(self, worker: QObject, finished, failed) -> None:
        self.worker_thread = QThread(self)
        self.worker = worker
        worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(worker.run)  # type: ignore[attr-defined]
        worker.finished.connect(finished)  # type: ignore[attr-defined]
        worker.failed.connect(failed)  # type: ignore[attr-defined]
        worker.finished.connect(self.worker_thread.quit)  # type: ignore[attr-defined]
        worker.failed.connect(self.worker_thread.quit)  # type: ignore[attr-defined]
        worker.finished.connect(worker.deleteLater)  # type: ignore[attr-defined]
        worker.failed.connect(worker.deleteLater)  # type: ignore[attr-defined]
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, "worker", None))
        self.worker_thread.start()

    @Slot(object)
    def on_capture_ready(self, capture: CaptureResult) -> None:
        self.refresh_capture_files(capture.image_path)
        self.image_label.setText(f"captured: {capture.image_path}")
        if capture.image_path.name.startswith("before_"):
            self.latest_before_path = capture.image_path
            self.set_status(AppState.IDLE, f"before captured {capture.image_path.name}; hover item and press F8")
            return
        if capture.image_path.name.startswith("after_"):
            if capture.before_image_path is not None:
                write_before_sidecar(capture.image_path, capture.before_image_path)
            before_message = f" using before {capture.before_image_path.name}" if capture.before_image_path else ""
            self.set_status(AppState.IDLE, f"after captured {capture.image_path.name}{before_message}; select PNG and analyze")
            return
        self.set_status(AppState.IDLE, f"captured {capture.image_path.name}; select PNG and analyze")

    @Slot(str)
    def on_capture_failed(self, message: str) -> None:
        self.set_status(AppState.ERROR, message)
        QMessageBox.warning(self, "Capture failed", message)
        self.set_status(AppState.IDLE, "F7 captures before. F8 captures after using latest F7.")

    @Slot(object)
    def on_analysis_ready(self, analysis: AnalysisResult) -> None:
        self.analysis = analysis
        self.set_status(AppState.REVIEWING, "analysis result - review and edit values")
        self._set_field("req_level", analysis.req_level.value)
        self._set_field("equipment_type", analysis.equipment_type.value)
        self._set_field("price_meso", analysis.price_meso.value)
        self._set_field("equipment_options", analysis.equipment_options.value)
        self._set_field("potential", analysis.potential.value)
        self.confidence_label.setText(
            "recognizer: opencv-template / confidence: "
            f"price={analysis.price_meso.confidence:.2f}, "
            f"options={analysis.equipment_options.confidence:.2f}, "
            f"potential={analysis.potential.confidence:.2f}"
        )
        self.training_label.setText(format_crop_preview_summary(build_crop_preview_summary(analysis)))
        self.populate_crop_preview(analysis)
        self.image_label.setText(f"image: {analysis.image_path}")
        self.show_before_preview_for_capture(analysis.image_path)
        self.show_diff_preview_for_capture(analysis.image_path)

    @Slot(str)
    def on_analysis_failed(self, message: str) -> None:
        self.set_status(AppState.ERROR, message)
        QMessageBox.warning(self, "Analyze failed", message)
        self.set_status(AppState.IDLE, "Select PNG and click Analyze.")

    def _set_field(self, key: str, value: object) -> None:
        widget = self.fields[key]
        text = "" if value is None else str(value)
        if isinstance(widget, QTextEdit):
            widget.setPlainText(text)
        else:
            widget.setText(text)

    def set_preview_pixmap(self, label: QLabel, image_path: Path, failure_text: str) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            if isinstance(label, ScaledImageLabel):
                label.clear_source(failure_text)
            else:
                label.clear()
                label.setText(failure_text)
            label.setMinimumSize(PREVIEW_SIZE)
            return
        if isinstance(label, ScaledImageLabel):
            label.set_source_pixmap(pixmap)
        else:
            scaled = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)

    def show_before_preview_for_capture(self, image_path: Path) -> None:
        self.set_preview_pixmap(self.preview_label, image_path, "Image load failed")

    def clear_diff_preview(self, text: str) -> None:
        self.diff_preview_label.clear_source(text)
        self.diff_preview_label.setMinimumSize(PREVIEW_SIZE)

    def clear_crop_preview(self, text: str = "") -> None:
        while self.crop_rows_layout.count():
            item = self.crop_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if text:
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.crop_rows_layout.addWidget(label)

    def populate_crop_preview(self, analysis: AnalysisResult) -> None:
        self.clear_crop_preview()
        rows = label_value_crop_rows(analysis)
        if not rows:
            self.clear_crop_preview("label/value crop 없음")
            return
        source = QPixmap(str(analysis.image_path))
        if source.isNull():
            self.clear_crop_preview("source image load failed")
            return
        previous_group = None
        for row in rows:
            group = crop_preview_group_label(row)
            if group != previous_group:
                self.crop_rows_layout.addWidget(crop_preview_group_header(group))
                previous_group = group
            self.crop_rows_layout.addWidget(self.make_crop_row_widget(source, row))

    def make_crop_row_widget(self, source: QPixmap, row: dict[str, object]) -> QWidget:
        if row.get("line_type") == "price" or row.get("sort_key") == "price":
            return self.make_price_crop_row_widget(source, row)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 6)
        title = QLabel(crop_row_title(row))
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)
        crop_row = QHBoxLayout()
        crop_row.addWidget(labeled_crop_widget("raw", crop_image_rect(source, row.get("raw_line_rect"), crop_fallback_text("raw"))))
        crop_row.addWidget(labeled_crop_widget("label", crop_image_rect(source, row.get("label_crop_rect"), crop_fallback_text("label"))))
        crop_row.addWidget(labeled_crop_widget("value", crop_image_rect(source, row.get("value_crop_rect"), crop_fallback_text("value"))))
        crop_row.addWidget(labeled_crop_widget("model", crop_image_label(source, row.get("model_trace"), crop_fallback_text("model"))))
        layout.addLayout(crop_row)
        detail = QLabel(crop_row_detail(row))
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(detail)
        widget.setStyleSheet(crop_row_style(row))
        return widget

    def make_price_crop_row_widget(self, source: QPixmap, row: dict[str, object]) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 6)
        title = QLabel(crop_row_title(row))
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)
        crop_row = QHBoxLayout()
        crop_row.addWidget(
            labeled_crop_widget(
                "price_search_roi",
                crop_image_path(
                    row.get("price_search_roi_path"),
                    crop_image_rect(source, row.get("raw_line_rect"), "price_search_roi: not available", PRICE_CROP_THUMB_MAX),
                    max_size=PRICE_CROP_THUMB_MAX,
                ),
            )
        )
        crop_row.addWidget(
            labeled_crop_widget(
                "price_tight_crop",
                crop_image_path(
                    row.get("price_tight_crop_path"),
                    crop_image_rect(source, row.get("value_crop_rect"), "price_tight_crop: not available", PRICE_CROP_THUMB_MAX),
                    max_size=PRICE_CROP_THUMB_MAX,
                ),
            )
        )
        crop_row.addWidget(
            labeled_crop_widget(
                "price_color_mask",
                crop_image_path(
                    row.get("price_color_mask_path"),
                    fallback_text="price_color_mask: not available",
                    max_size=PRICE_CROP_THUMB_MAX,
                ),
            )
        )
        crop_row.addWidget(
            labeled_crop_widget(
                "price_component_mask",
                crop_image_path(
                    row.get("price_component_mask_path"),
                    fallback_text="price_component_mask: not available",
                    max_size=PRICE_CROP_THUMB_MAX,
                ),
            )
        )
        layout.addLayout(crop_row)
        detail = QLabel(crop_row_detail(row))
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(detail)
        widget.setStyleSheet(crop_row_style(row))
        return widget

    def show_diff_preview_for_capture(self, image_path: Path) -> None:
        diff_path = self.diff_preview_path(image_path)
        if diff_path is None:
            self.clear_diff_preview("diff 없음\n(before 연결 후 Analyze 필요)")
            return
        self.set_preview_pixmap(self.diff_preview_label, diff_path, "DIFF load failed")

    def diff_preview_path(self, image_path: Path) -> Path | None:
        diff_dir = self.config.capture.debug_dir / "diff"
        stem = image_path.stem
        for suffix in ("analysis_binary", "residual", "final_mask", "foreground_color_final", "foreground_text_mask"):
            candidate = diff_dir / f"{stem}_{suffix}.png"
            if candidate.exists():
                return candidate
        return None

    def _field_value(self, key: str) -> str:
        widget = self.fields[key]
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        return widget.text()

    def current_values(self) -> dict[str, object]:
        return {
            "req_level": parse_required_int(self._field_value("req_level")),
            "equipment_type": self._field_value("equipment_type").strip(),
            "price_meso": parse_required_int(self._field_value("price_meso")),
            "price_meso_text": self._field_value("price_meso").strip(),
            "str_value": 0,
            "dex_value": 0,
            "int_value": 0,
            "luk_value": 0,
            "attack": 0,
            "magic_attack": 0,
            "upgrade_count": 0,
            "black_crystal": "",
            "equipment_options": self._field_value("equipment_options").strip(),
            "potential": self._field_value("potential").strip(),
        }

    def save_current(self) -> None:
        if self.analysis is None:
            return
        missing = self.missing_required_fields()
        if missing:
            message = "Required fields are empty: " + ", ".join(missing)
            self.set_status(AppState.REVIEWING, message)
            QMessageBox.warning(self, "Save blocked", message)
            return
        values = self.current_values()
        try:
            self.set_status(AppState.SAVING, "saving")
            record = final_record_from_analysis(self.analysis, values)
            if self.storage.has_recent_duplicate(record, self.config.duplicate_window_seconds):
                result = QMessageBox.question(
                    self,
                    "Duplicate warning",
                    "A similar record was saved recently. Save anyway?",
                )
                if result != QMessageBox.StandardButton.Yes:
                    self.set_status(AppState.REVIEWING, "save cancelled")
                    return
            record_id = self.storage.save(record)
        except Exception as exc:
            logger.exception("save failed")
            self.set_status(AppState.ERROR, str(exc))
            QMessageBox.warning(self, "Save failed", str(exc))
            return

        sample_message = ""
        if self.config.vision.save_training_samples:
            try:
                summary = TrainingSampleWriter(self.config.vision).save_confirmed_samples(self.analysis, values)
                sample_message = "; db saved, " + format_sample_save_summary(summary)
                if summary.skipped_count:
                    reasons = ", ".join(sorted(set(summary.skipped_reasons))) or "unknown"
                    sample_message += f" skipped={summary.skipped_count} ({reasons})"
                if summary.errors:
                    logger.warning("training sample save errors: %s", summary.errors)
                    QMessageBox.warning(self, "Training sample warning", "\n".join(summary.errors))
            except Exception as exc:
                logger.exception("training sample save failed")
                sample_message = "; training sample save failed"
                QMessageBox.warning(self, "Training sample warning", str(exc))

        self.set_status(AppState.IDLE, f"saved record #{record_id}{sample_message}; F7 before / F8 after")
        self.analysis = None

    def missing_required_fields(self) -> list[str]:
        required = [
            "req_level",
            "equipment_type",
            "price_meso",
            "equipment_options",
            "potential",
        ]
        missing = []
        for key in required:
            value = self._field_value(key).strip()
            if value == "" or value.lower() == "none":
                missing.append(key)
        return missing

    def cancel_review(self) -> None:
        self.analysis = None
        self.clear_crop_preview("Analyze 후 crop 표시")
        self.set_status(AppState.IDLE, "cancelled; select PNG and click Analyze")

    def open_capture_settings(self) -> None:
        dialog = CaptureSettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply_to_config()
            self.set_status(
                AppState.IDLE,
                (
                    "capture region updated: "
                    f"L{self.config.capture.left} R{self.config.capture.right} "
                    f"U{self.config.capture.up} D{self.config.capture.down}"
                ),
            )


class CaptureSettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Capture Region")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.left = self._spin(config.capture.left)
        self.right = self._spin(config.capture.right)
        self.up = self._spin(config.capture.up)
        self.down = self._spin(config.capture.down)
        form.addRow("Left px", self.left)
        form.addRow("Right px", self.right)
        form.addRow("Up px", self.up)
        form.addRow("Down px", self.down)
        layout.addLayout(form)

        self.apply_button = QPushButton("Apply Runtime")
        self.save_button = QPushButton("Save to config.yaml")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)
        self.save_button.clicked.connect(self.save_and_accept)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)
        layout.addWidget(buttons)

    def _spin(self, value: int) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(0, 3000)
        spin.setSingleStep(10)
        spin.setValue(value)
        return spin

    def apply_to_config(self) -> None:
        self.config.capture.left = self.left.value()
        self.config.capture.right = self.right.value()
        self.config.capture.up = self.up.value()
        self.config.capture.down = self.down.value()

    def save_and_accept(self) -> None:
        self.apply_to_config()
        save_config(self.config)
        self.accept()


def parse_required_int(text: str) -> int:
    normalized = text.strip().replace(",", "")
    if normalized == "" or normalized.lower() == "none":
        raise ValueError("required integer field is empty")
    return int(normalized)


def format_label_value_preview(analysis: AnalysisResult) -> str:
    rows = label_value_crop_rows(analysis)
    if not rows:
        return "label/value trace 없음"
    lines = []
    for row in rows:
        prefix = f"line {row['line_index']}" if row["line_index"] is not None else str(row.get("sort_key", "trace"))
        label = row["label"] or "-"
        value = row["value"] or "-"
        line_type = f" / type={row['line_type']}" if row.get("line_type") else ""
        field = f" / field={row['field_name']}" if row.get("field_name") else ""
        text = f" / text={row['line_text']}" if row["line_text"] else ""
        reason = f" / reason={row['reason']}" if row["reason"] else ""
        notes = f" / {'; '.join(dict.fromkeys(row['notes']))}" if row["notes"] else ""
        lines.append(f"{prefix}: label={label} / value={value} / status={row['status']}{line_type}{field}{text}{reason}{notes}")
    return "\n".join(lines)


def label_value_crop_rows(analysis: AnalysisResult) -> list[dict[str, object]]:
    traces = deepcopy(analysis.traces)
    rows: dict[object, dict[str, object]] = {}
    for trace in traces:
        if trace.field_type not in {"item_metadata", "option_label", "option_value", "price", "rejected", "ignored", "ui_label", "ui_value"}:
            continue
        key = crop_row_key(trace)
        row = rows.setdefault(
            key,
            {
                "sort_key": key,
                "line_index": trace.line_index,
                "line_text": "",
                "label": "",
                "value": "",
                "field_name": "",
                "line_type": "",
                "parsed_key": "",
                "parsed_value": "",
                "raw_prediction": "",
                "selected_prediction": "",
                "confidence": 0.0,
                "semantic_validation_status": "",
                "semantic_validation_reason": "",
                "review_status": "unreviewed",
                "status": "ok",
                "reason": "",
                "notes": [],
                "raw_line_rect": None,
                "label_crop_rect": None,
                "value_crop_rect": None,
                "label_trace": None,
                "value_trace": None,
                "model_trace": None,
                "price_search_roi_path": "",
                "price_tight_crop_path": "",
                "price_color_mask_path": "",
                "price_component_mask_path": "",
            },
        )
        metadata = trace.crop_metadata or {}
        line_text = str(metadata.get("line_text") or metadata.get("parsed_line_text") or "")
        if line_text and not row["line_text"]:
            row["line_text"] = line_text
        if not row["field_name"]:
            row["field_name"] = trace.field_name
        if not row["line_type"]:
            row["line_type"] = str(metadata.get("line_type") or line_type_for_trace(trace))
        if not row["parsed_key"]:
            row["parsed_key"] = str(metadata.get("metadata_key") or metadata.get("parsed_option_key") or "")
        if not row["parsed_value"]:
            row["parsed_value"] = str(metadata.get("parsed_value_text") or trace.selected_prediction or "")
        if not row["raw_prediction"]:
            row["raw_prediction"] = str(trace.raw_prediction or "")
        if not row["selected_prediction"]:
            row["selected_prediction"] = str(trace.selected_prediction or "")
        if trace.confidence and float(trace.confidence) > float(row["confidence"]):
            row["confidence"] = float(trace.confidence)
        if not row["semantic_validation_status"]:
            row["semantic_validation_status"] = str(metadata.get("semantic_validation_status") or "")
        if not row["semantic_validation_reason"]:
            row["semantic_validation_reason"] = str(metadata.get("semantic_validation_reason") or "")
        if metadata.get("review_status"):
            row["review_status"] = str(metadata.get("review_status"))
        for path_key in ("price_search_roi_path", "price_tight_crop_path", "price_color_mask_path", "price_component_mask_path"):
            if metadata.get(path_key) and not row[path_key]:
                row[path_key] = str(metadata.get(path_key))
        if row["raw_line_rect"] is None:
            row["raw_line_rect"] = rect_from_metadata(metadata.get("raw_line_rect") or metadata.get("price_search_rect") or metadata.get("search_rect"))
        if row["label_crop_rect"] is None:
            row["label_crop_rect"] = rect_from_metadata(metadata.get("label_crop_rect") or metadata.get("trimmed_label_rect"))
        if row["value_crop_rect"] is None:
            row["value_crop_rect"] = rect_from_metadata(
                metadata.get("value_crop_rect")
                or metadata.get("price_tight_rect")
                or metadata.get("tight_rect")
                or metadata.get("raw_value_rect")
            )
        reason = str(metadata.get("rejection_reason") or "")
        if trace.field_type == "ignored" or row["line_type"] == "ignored":
            row["status"] = "ignored"
            row["reason"] = reason or str(metadata.get("ignored_reason") or trace.selection_reason or "ignored")
        elif trace.field_type == "rejected" or reason:
            row["status"] = "rejected"
            row["reason"] = reason or "rejected"
        elif trace.needs_review and row["status"] != "rejected":
            row["status"] = "review"
        if metadata.get("line_order_corrected"):
            original_key = metadata.get("original_parsed_option_key")
            original_value = metadata.get("original_parsed_value_text")
            note_parts = []
            if original_key:
                note_parts.append(f"label:{original_key}")
            if original_value:
                note_parts.append(f"value:{original_value}")
            if note_parts:
                row["notes"].append("corrected from " + ", ".join(note_parts))
        if row["line_type"] == "price" or trace.field_name == "price_meso":
            row["label"] = "price"
            row["value"] = display_value_for_trace(trace)
            row["value_trace"] = trace
            row["model_trace"] = trace
        elif trace.field_type == "item_metadata":
            row["label"] = str(metadata.get("metadata_key") or trace.field_name)
            row["value"] = display_value_for_trace(trace)
            row["value_trace"] = trace
            row["model_trace"] = trace
        elif trace.field_name.endswith("_label") or trace.field_type in {"option_label", "ui_label"}:
            row["label"] = display_label_for_trace(trace)
            row["label_trace"] = trace
            if row["model_trace"] is None:
                row["model_trace"] = trace
        elif trace.field_type in {"option_value", "ui_value"} or trace.field_type == "rejected":
            row["value"] = display_value_for_trace(trace)
            row["value_trace"] = trace
            row["model_trace"] = trace
    return sorted(rows.values(), key=sort_preview_row_key)


def crop_row_key(trace) -> object:
    if trace.field_type == "item_metadata":
        return f"metadata:{trace.crop_metadata.get('metadata_key') or trace.field_name}"
    if trace.field_type == "price" or trace.field_name == "price_meso" or (trace.crop_metadata or {}).get("line_type") == "price":
        return "price"
    if trace.field_name in {"req_level", "req_level_label"}:
        return "req_level"
    return trace.line_index if trace.line_index is not None else trace.field_name


def crop_row_title(row: dict[str, object]) -> str:
    prefix = f"line {row['line_index']}" if row["line_index"] is not None else str(row.get("sort_key", "trace"))
    label = row.get("label") or "-"
    value = row.get("value") or "-"
    status = row.get("status") or "ok"
    reason = f" / {row['reason']}" if row.get("reason") else ""
    line_type = row.get("line_type") or "-"
    field_name = row.get("field_name") or "-"
    validation = row.get("semantic_validation_status") or row.get("review_status") or ""
    validation_text = f" / validation={validation}" if validation else ""
    return f"{prefix}  {line_type} / {field_name}  label={label}  value={value}  {status}{validation_text}{reason}"


def crop_row_detail(row: dict[str, object]) -> str:
    parts = [
        f"parsed_key={row.get('parsed_key') or '-'}",
        f"parsed_value={row.get('parsed_value') or '-'}",
        f"raw={row.get('raw_prediction') or '-'}",
        f"selected={row.get('selected_prediction') or '-'}",
        f"confidence={float(row.get('confidence') or 0.0):.2f}",
        f"review={row.get('review_status') or 'unreviewed'}",
    ]
    if row.get("semantic_validation_status"):
        parts.append(f"validation={row.get('semantic_validation_status')}")
    if row.get("semantic_validation_reason"):
        parts.append(f"validation_reason={row.get('semantic_validation_reason')}")
    if row.get("reason"):
        parts.append(f"reason={row.get('reason')}")
    if row.get("line_text"):
        parts.append(f"text={row.get('line_text')}")
    return " / ".join(parts)


def crop_row_style(row: dict[str, object]) -> str:
    status = str(row.get("status") or "")
    reason = str(row.get("reason") or row.get("semantic_validation_reason") or "")
    review_status = str(row.get("review_status") or "")
    if status == "rejected" or "failed" in str(row.get("semantic_validation_status") or ""):
        background = "#fff0f0"
        border = "#d43f3a"
    elif status == "ignored":
        background = "#f3f4f6"
        border = "#9aa5b1"
    elif "split_uncertain" in reason:
        background = "#fff7e6"
        border = "#d9822b"
    elif review_status in {"unreviewed", "pending_review"} or status == "review":
        background = "#fffbe6"
        border = "#d9b600"
    else:
        background = "#ffffff"
        border = "#d0d0d0"
    return f"QWidget {{ background: {background}; border-bottom: 1px solid {border}; }} QLabel {{ border: 0; }}"


def crop_preview_group_header(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        "QLabel { background: #263238; color: #ffffff; font-weight: bold; "
        "padding: 5px 8px; border: 0; }"
    )
    return label


def labeled_crop_widget(title: str, image_label: QLabel) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    caption = QLabel(title)
    caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(caption)
    layout.addWidget(image_label)
    return widget


def crop_fallback_text(slot: str) -> str:
    return {
        "raw": "raw line crop: not saved",
        "label": "label crop: not available",
        "value": "value crop: split failed",
        "model": "model crop: not generated",
    }.get(slot, "crop: not available")


def crop_image_label(source: QPixmap, trace: object, fallback: str) -> QLabel:
    return crop_image_rect(source, getattr(trace, "crop_rect", None), fallback)


def crop_image_rect(source: QPixmap, rect: Rect | None, fallback: str, max_size: QSize = CROP_THUMB_MAX) -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(max_size)
    label.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #777; }")
    if rect is None or rect.width <= 0 or rect.height <= 0:
        label.setText(fallback)
        return label
    crop = source.copy(rect.left, rect.top, rect.width, rect.height)
    if crop.isNull():
        label.setText("crop load failed")
        return label
    scaled = crop.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
    label.setPixmap(scaled)
    return label


def crop_image_path(
    path_value: object,
    fallback: QLabel | None = None,
    fallback_text: str = "crop: not available",
    max_size: QSize = CROP_THUMB_MAX,
) -> QLabel:
    if path_value:
        pixmap = QPixmap(str(path_value))
        if not pixmap.isNull():
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(max_size)
            label.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #777; }")
            label.setPixmap(pixmap.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
            return label
    if fallback is not None:
        return fallback
    label = QLabel(fallback_text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(max_size)
    label.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #777; }")
    return label


def rect_from_metadata(value: object) -> Rect | None:
    if not isinstance(value, dict):
        return None
    try:
        return Rect(int(value["left"]), int(value["top"]), int(value["right"]), int(value["bottom"]))
    except (KeyError, TypeError, ValueError):
        return None


def line_type_for_trace(trace) -> str:
    if trace.field_type == "item_metadata":
        key = (trace.crop_metadata or {}).get("metadata_key") or trace.field_name
        return "metadata_req_level" if key == "req_level" else "metadata_equipment_category"
    if trace.field_type == "option_label" or trace.field_type == "option_value":
        return "potential_option" if str(trace.field_name).startswith("potential_") else "base_option"
    if trace.field_type == "price":
        return "price"
    if trace.field_type == "rejected":
        return str((trace.crop_metadata or {}).get("line_type") or "rejected")
    if trace.field_type == "ignored":
        return "ignored"
    return str(trace.field_type or "")


def build_crop_preview_summary(analysis: AnalysisResult) -> dict[str, int]:
    summary = {
        "item_metadata": 0,
        "option_label": 0,
        "option_value": 0,
        "price": 0,
        "ignored": 0,
        "rejected": 0,
        "split_uncertain": 0,
    }
    seen: set[tuple[str, object]] = set()
    for trace in analysis.traces:
        metadata = trace.crop_metadata or {}
        reason = str(metadata.get("rejection_reason") or trace.selection_reason or "")
        if "split_uncertain" in reason or "split_uncertain" in str(metadata.get("semantic_validation_reason") or ""):
            summary["split_uncertain"] += 1
        if trace.field_type == "ignored" or metadata.get("line_type") == "ignored":
            summary["ignored"] += 1
            continue
        if trace.field_type == "rejected" or metadata.get("rejection_reason"):
            summary["rejected"] += 1
            continue
        if trace.field_type in {"item_metadata", "option_label", "option_value", "price"} and trace.crop_rect is not None:
            key = (trace.field_type, trace.field_name, trace.line_index)
            if key in seen:
                continue
            seen.add(key)
            summary[trace.field_type] += 1
    return summary


def format_crop_preview_summary(summary: dict[str, int]) -> str:
    return (
        "Training sample preview: "
        f"item_metadata={summary.get('item_metadata', 0)}, "
        f"option_label={summary.get('option_label', 0)}, "
        f"option_value={summary.get('option_value', 0)}, "
        f"price={summary.get('price', 0)}, "
        f"ignored={summary.get('ignored', 0)}, "
        f"rejected={summary.get('rejected', 0)}, "
        f"split_uncertain={summary.get('split_uncertain', 0)}"
    )


def format_sample_save_summary(summary: SampleSaveSummary) -> str:
    return (
        "training samples: "
        f"item_metadata saved={summary.item_metadata_count}, "
        f"option_label saved={summary.option_label_count}, "
        f"option_value saved={summary.option_value_count}, "
        f"price saved={summary.price_count}, "
        f"rejected={summary.rejected_count}"
    )


def display_label_for_trace(trace) -> str:
    metadata = trace.crop_metadata or {}
    return str(
        metadata.get("confirmed_option_key")
        or metadata.get("parsed_option_key")
        or trace.selected_prediction
        or trace.raw_prediction
        or ""
    )


def display_value_for_trace(trace) -> str:
    metadata = trace.crop_metadata or {}
    if metadata.get("line_type") == "price" or trace.field_name == "price_meso":
        if trace.field_type == "rejected" or metadata.get("rejection_reason"):
            return str(trace.selected_prediction or "")
        return str(trace.selected_prediction or metadata.get("parsed_value_text") or "")
    return str(
        metadata.get("confirmed_value_text")
        or metadata.get("parsed_value_text")
        or metadata.get("label")
        or trace.selected_prediction
        or trace.raw_prediction
        or ""
    )


def sort_preview_row_key(row: dict[str, object]) -> tuple[int, str, str]:
    group_order = {
        "req_level": 0,
        "equipment_category": 1,
        "price": 2,
        "equipment_options": 3,
        "potential": 4,
        "other": 5,
    }
    group = crop_preview_group_key(row)
    line_index = row.get("line_index")
    if isinstance(line_index, int):
        position = f"{line_index:06d}"
    else:
        position = str(row.get("sort_key") or row.get("field_name") or "")
    return (group_order.get(group, 5), position, str(row.get("field_name") or ""))


def crop_preview_group_key(row: dict[str, object]) -> str:
    field_name = str(row.get("field_name") or "")
    line_type = str(row.get("line_type") or "")
    sort_key = str(row.get("sort_key") or "")
    label = str(row.get("label") or "")
    if sort_key == "req_level" or field_name in {"req_level", "req_level_label"} or line_type == "metadata_req_level":
        return "req_level"
    if (
        sort_key == "metadata:equipment_category"
        or field_name == "equipment_category"
        or line_type == "metadata_equipment_category"
        or label == "equipment_category"
    ):
        return "equipment_category"
    if sort_key == "price" or field_name == "price" or line_type == "price":
        return "price"
    if line_type == "potential_option" or field_name.startswith("potential_"):
        return "potential"
    if line_type == "base_option" or field_name.endswith("_label") or row.get("label_trace") is not None or row.get("value_trace") is not None:
        return "equipment_options"
    return "other"


def crop_preview_group_label(row: dict[str, object]) -> str:
    return {
        "req_level": "REQ LEV",
        "equipment_category": "장비분류",
        "price": "가격",
        "equipment_options": "장비옵션",
        "potential": "잠재능력",
    }.get(crop_preview_group_key(row), "기타")


def parse_optional_int(text: str) -> int:
    normalized = text.strip().replace(",", "")
    if normalized == "" or normalized.lower() == "none":
        return 0
    return int(normalized)


def before_sidecar_path(after_image_path: Path) -> Path:
    return after_image_path.with_suffix(after_image_path.suffix + ".before.txt")


def write_before_sidecar(after_image_path: Path, before_image_path: Path) -> None:
    capture_pair_id = capture_pair_id_from_path(before_image_path)
    before_sidecar_path(after_image_path).write_text(
        sidecar_payload(before_image_path, capture_pair_id, session_id_from_pair_id(capture_pair_id)),
        encoding="utf-8",
    )


def read_before_sidecar(after_image_path: Path) -> Path | None:
    sidecar = before_sidecar_path(after_image_path)
    if not sidecar.exists():
        return None
    try:
        payload = parse_sidecar(sidecar.read_text(encoding="utf-8"))
    except OSError:
        return None
    value = payload.get("before_image_path", "")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def read_capture_pair_id_sidecar(after_image_path: Path) -> str:
    sidecar = before_sidecar_path(after_image_path)
    if not sidecar.exists():
        return ""
    try:
        return parse_sidecar(sidecar.read_text(encoding="utf-8")).get("capture_pair_id", "")
    except OSError:
        return ""


def read_session_id_sidecar(after_image_path: Path) -> str:
    sidecar = before_sidecar_path(after_image_path)
    if not sidecar.exists():
        return ""
    try:
        return parse_sidecar(sidecar.read_text(encoding="utf-8")).get("session_id", "")
    except OSError:
        return ""


class _F8Event(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self) -> None:
        super().__init__(self.EVENT_TYPE)


class _F7Event(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self) -> None:
        super().__init__(self.EVENT_TYPE)


def run_app(config: AppConfig, storage: Storage) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = ReviewWindow(config, storage)
    window.show()
    return app.exec()

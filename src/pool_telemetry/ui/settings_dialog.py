from __future__ import annotations

from typing import Optional

from PyQt6 import QtWidgets

from ..config import AppConfig, save_config


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: AppConfig, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("Settings")
        self.setModal(True)

        self._gemini_key = QtWidgets.QLineEdit()
        self._gemini_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._gemini_key.setText(config.api_keys.gemini or "")

        self._anthropic_key = QtWidgets.QLineEdit()
        self._anthropic_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._anthropic_key.setText(config.api_keys.anthropic or "")

        self._data_dir = QtWidgets.QLineEdit()
        self._data_dir.setText(config.storage.data_directory)

        self._warn_threshold = QtWidgets.QDoubleSpinBox()
        self._warn_threshold.setRange(0, 1000)
        self._warn_threshold.setValue(config.cost_tracking.warn_threshold_usd)

        self._stop_threshold = QtWidgets.QDoubleSpinBox()
        self._stop_threshold.setRange(0, 1000)
        self._stop_threshold.setValue(config.cost_tracking.stop_threshold_usd or 0.0)
        self._stop_threshold.setSpecialValueText("No limit")

        form = QtWidgets.QFormLayout()
        form.addRow("Gemini API Key", self._gemini_key)
        form.addRow("Anthropic API Key", self._anthropic_key)
        form.addRow("Data Directory", self._data_dir)
        form.addRow("Warn Threshold (USD)", self._warn_threshold)
        form.addRow("Stop Threshold (USD)", self._stop_threshold)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_data_dir)
        form.addRow("", browse_button)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse_data_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self._data_dir.text(),
        )
        if path:
            self._data_dir.setText(path)

    def _apply_config(self) -> None:
        self._config.api_keys.gemini = self._gemini_key.text().strip() or None
        self._config.api_keys.anthropic = self._anthropic_key.text().strip() or None
        self._config.storage.data_directory = self._data_dir.text().strip()
        self._config.cost_tracking.warn_threshold_usd = float(self._warn_threshold.value())
        stop_value = float(self._stop_threshold.value())
        self._config.cost_tracking.stop_threshold_usd = stop_value if stop_value > 0 else None
        save_config(self._config)

    def _on_apply(self) -> None:
        self._apply_config()

    def _on_accept(self) -> None:
        self._apply_config()
        self.accept()

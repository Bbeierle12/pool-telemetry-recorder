from __future__ import annotations


from PyQt6 import QtWidgets


class GoProConnectDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect GoPro")
        self.setModal(True)
        self._device_index = QtWidgets.QSpinBox()
        self._device_index.setRange(0, 10)
        self._device_index.setValue(0)

        self._resolution = QtWidgets.QComboBox()
        self._resolution.addItems(["1080p", "720p", "4k"])

        self._framerate = QtWidgets.QComboBox()
        self._framerate.addItems(["24", "30", "60", "120"])
        self._framerate.setCurrentText("60")

        form = QtWidgets.QFormLayout()
        form.addRow("Device index", self._device_index)
        form.addRow("Resolution", self._resolution)
        form.addRow("Framerate", self._framerate)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple[int, str, float]:
        return (
            int(self._device_index.value()),
            self._resolution.currentText(),
            float(self._framerate.currentText()),
        )

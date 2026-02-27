from PyQt6.QtWidgets import QFrame, QGridLayout


class CardPanel(QFrame):
    def __init__(self, parent=None, margins=(20, 20, 20, 20), spacing=12):
        super().__init__(parent)
        self.setObjectName("cardPanel")
        self.setFrameStyle(QFrame.Shape.NoFrame)
        layout = QGridLayout(self)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)

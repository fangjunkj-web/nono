import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from covermatcher.ui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('AI Cover Matcher')
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

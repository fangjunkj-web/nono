from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import *
from .io import load_topics, list_images
from .matcher import Matcher
from .render import render


class Worker(QThread):
    done = Signal(object)
    status = Signal(str)
    prog = Signal(int)

    def __init__(self, titles, images):
        super().__init__()
        self.titles = titles
        self.images = images

    def run(self):
        try:
            cache = Path.home() / 'Library' / 'Caches' / 'AI Cover Matcher' / 'embeddings'
            self.status.emit('Loading local vision model…')
            m = Matcher(cache)

            def cb(a, b):
                self.prog.emit(int(a * 100 / max(b, 1)))
                if b == 1:
                    self.status.emit('Library index found. Reusing previous analysis…')
                else:
                    self.status.emit(f'Analyzing new or changed images {a}/{b}…')

            result, changed, valid = m.match(self.titles, self.images, cb, 5)
            self.done.emit({'matches': result, 'changed': changed, 'valid': valid})
        except Exception as e:
            self.done.emit(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Cover Matcher')
        self.resize(1280, 820)
        self.topics = []
        self.images = []
        self.matches = []
        self.logo = ''
        self.library_path = ''

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        h = QHBoxLayout()
        v.addLayout(h)

        for label, fn in [
            ('Import Excel / CSV', self.pick_topics),
            ('Choose Image Library', self.pick_library),
            ('Choose / Replace Logo', self.pick_logo),
            ('Auto Match', self.do_match),
            ('Export All Covers', self.export_all),
        ]:
            b = QPushButton(label)
            b.clicked.connect(fn)
            h.addWidget(b)

        self.info = QLabel('Import topics and choose an image library. Imported libraries are indexed once and reused.')
        v.addWidget(self.info)
        self.progress = QProgressBar()
        self.progress.hide()
        v.addWidget(self.progress)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['#', 'Article title', 'Matched image', 'Score'])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        v.addWidget(self.table)

        self.preview = QLabel('Preview')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(270)
        v.addWidget(self.preview)
        self.table.currentCellChanged.connect(self.show_preview)

    def pick_topics(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Topics', '', 'Excel/CSV (*.xlsx *.xlsm *.csv)')
        if p:
            self.topics = load_topics(p)
            self.info.setText(f'{len(self.topics)} topics loaded.')

    def pick_library(self):
        p = QFileDialog.getExistingDirectory(self, 'Image library')
        if p:
            self.library_path = p
            self.info.setText('Scanning filenames only…')
            QApplication.processEvents()
            self.images = list_images(p)
            self.info.setText(f'{len(self.images)} images found. Existing index will be reused; only new/changed files are analyzed.')

    def pick_logo(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Logo', '', 'Images (*.png *.jpg *.jpeg *.webp)')
        if p:
            self.logo = p
            self.info.setText('Logo selected. You can replace it anytime.')

    def do_match(self):
        if not self.topics or not self.images:
            return QMessageBox.warning(self, 'Missing input', 'Import topics and choose an image library first.')
        self.progress.setValue(0)
        self.progress.show()
        self.worker = Worker(self.topics, self.images)
        self.worker.prog.connect(self.progress.setValue)
        self.worker.status.connect(self.info.setText)
        self.worker.done.connect(self.match_done)
        self.worker.start()

    def match_done(self, r):
        self.progress.hide()
        if isinstance(r, Exception):
            return QMessageBox.critical(self, 'Matching failed', str(r))

        self.matches = r['matches']
        changed = r['changed']
        valid = r['valid']
        self.table.setRowCount(len(self.topics))

        for i, (t, cands) in enumerate(zip(self.topics, self.matches)):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(t))
            if cands:
                self.table.setItem(i, 3, QTableWidgetItem(f'{cands[0][1] * 100:.1f}%'))
                combo = QComboBox()
                for k, (p, s) in enumerate(cands):
                    combo.addItem(f'{k + 1}. {p.name} ({s * 100:.1f}%)', str(p))
                combo.currentIndexChanged.connect(lambda _, row=i, c=combo: self.change_pick(row, c))
                self.table.setCellWidget(i, 2, combo)

        if changed == 0:
            self.info.setText(f'Matching complete. Reused saved index for {valid} images; no image re-analysis needed.')
        else:
            self.info.setText(f'Matching complete. Library has {valid} readable images; analyzed only {changed} new/changed candidates.')

    def change_pick(self, row, combo):
        self.table.setItem(row, 3, QTableWidgetItem('selected'))
        self.show_preview(row, 0, -1, -1)

    def selected_path(self, row):
        w = self.table.cellWidget(row, 2)
        return Path(w.currentData()) if w else self.matches[row][0][0]

    def show_preview(self, row, *_):
        if row < 0 or not self.matches or row >= len(self.matches) or not self.matches[row]:
            return
        tmp = Path.home() / 'Library' / 'Caches' / 'AI Cover Matcher' / 'preview.jpg'
        try:
            render(self.selected_path(row), self.topics[row], tmp, self.logo or None)
            self.preview.setPixmap(QPixmap(str(tmp)).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.info.setText(f'Preview skipped: {e}')

    def export_all(self):
        if not self.matches:
            return QMessageBox.warning(self, 'Nothing to export', 'Run Auto Match first.')
        out = QFileDialog.getExistingDirectory(self, 'Export folder')
        if not out:
            return
        exported = 0
        skipped = 0
        for i, t in enumerate(self.topics):
            if not self.matches[i]:
                skipped += 1
                continue
            try:
                render(self.selected_path(i), t, Path(out) / f'{i + 1:03d}_{safe(t)}.jpg', self.logo or None)
                exported += 1
            except Exception:
                skipped += 1
        QMessageBox.information(self, 'Done', f'Exported {exported} covers. Skipped {skipped}.')


def safe(s):
    return ''.join(c if c.isalnum() or c in '-_ ' else '' for c in s).strip().replace(' ', '-')[:120]

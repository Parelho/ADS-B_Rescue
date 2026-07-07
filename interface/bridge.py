from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class Bridge(QObject):

    planeClicked = pyqtSignal(str, float, float)

    @pyqtSlot(str, str, float, float)
    def onPlaneClicked(self, callsign, country, lat, lon):
        self.planeClicked.emit(callsign, lat, lon)
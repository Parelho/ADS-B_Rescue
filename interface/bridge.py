from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class Bridge(QObject):

    planeClicked = pyqtSignal(str, float, float, str)

    @pyqtSlot(str, str, float, float, str)
    def onPlaneClicked(self, callsign, country, lat, lon, time):
        self.planeClicked.emit(callsign, lat, lon, time)
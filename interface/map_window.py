import io
import html
import json

import folium
from folium import Element

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QTextEdit,
    QHBoxLayout,
)

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from bridge import Bridge
from planes import get_lista

class MapWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aviões em tempo real (OpenSky)")
        self.setGeometry(100, 100, 1100, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        self.info_panel = QTextEdit()
        self.info_panel.setFixedWidth(320)
        self.info_panel.setReadOnly(True)
        self.info_panel.setPlainText("Clique em um marcador para ver os detalhes do voo.")
        self.info_panel.setStyleSheet(
            "QTextEdit {"
            "background: #f4f6f8;"
            "border: 1px solid #d1d5db;"
            "border-radius: 10px;"
            "padding: 12px;"
            "font-family: Arial, sans-serif;"
            "font-size: 12pt;"
            "color: #1f2937;"
            "}" 
        )

        self.view = QWebEngineView()

        layout.addWidget(self.view)
        layout.addWidget(self.info_panel)


        self.bridge = Bridge()
        self.bridge.planeClicked.connect(self.update_panel)

        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        self.update_map()

    def update_panel(self, callsign, lat, lon):
        safe_callsign = html.unescape(callsign).replace('<div>', '').replace('</div>', '')
        self.info_panel.setHtml(
            f"<div style='font-size:14pt; font-weight:bold; margin-bottom:8px;'>{safe_callsign}</div>"
            f"<hr style='border:none; border-top:1px solid #d1d5db; margin:0 0 12px 0;'/>"
            f"<div style='font-size:11pt; line-height:1.5;'>"
            f"<b>Latitude:</b> {lat}<br/>"
            f"<b>Longitude:</b> {lon}"
            f"</div>"
        )

    def update_map(self):

        m = folium.Map(
            location=[-14.235, -51.925],
            zoom_start=4,
            tiles="CartoDB positron"
        )

        planes = get_lista()

        for p in planes:
            folium.Marker(
                location=[p["trajectory"][0]["lat"], p["trajectory"][0]["lon"]],
                tooltip=p["icao"],
                icon=folium.Icon(
                    color="red",
                    icon="plane",
                    prefix="fa"
                ),
            ).add_to(m)

        plane_data = [
            {
                "callsign": plane["callsign"],
                "lat": plane["trajectory"][0]["lat"],
                "lon": plane["trajectory"][0]["lon"],
                "route_points": [[point["lat"], point["lon"]] for point in plane.get("trajectory", [])],
            }
            for plane in planes
        ]

        channel_js = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.bridge = channel.objects.bridge;
        });
        </script>
        """

        m.get_root().html.add_child(Element(channel_js))

        click_js = """
        <script>
        let planeData = """ + json.dumps(plane_data) + """;
        let currentLine = null;

        function normalizeText(value) {
            return String(value || "")
                .replace(/<[^>]*>/g, "")
                .replace(/&nbsp;/g, " ")
                .trim();
        }

        function findPlaneMatch(tooltip, latlng) {
            const normalizedTooltip = normalizeText(tooltip);

            const byCallsign = planeData.find(function(item) {
                return normalizeText(item.callsign) === normalizedTooltip;
            });

            if (byCallsign) {
                return byCallsign;
            }

            return planeData.find(function(item) {
                return Math.abs(item.lat - latlng.lat) < 0.0001 && Math.abs(item.lon - latlng.lng) < 0.0001;
            }) || null;
        }

        function attachClicks(){

            for(let i in window){

                try{

                    let obj = window[i];

                    if(obj && obj._latlng){

                        obj.on('click', function(e){

                            let p = e.target;
                            let map = p._map;
                            let latlng = p.getLatLng();
                            let tooltip = p.getTooltip() ? p.getTooltip().getContent() : "N/A";
                            let matchedPlane = findPlaneMatch(tooltip, latlng);
                            let routePoints = matchedPlane ? matchedPlane.route_points : [];

                            if (map) {
                                if (currentLine) {
                                    map.removeLayer(currentLine);
                                }

                                currentLine = L.polyline(
                                    routePoints.concat([[latlng.lat, latlng.lng]]),
                                    {
                                        color: 'blue',
                                        weight: 2,
                                        opacity: 0.8
                                    }
                                ).addTo(map);

                                if (map.getZoom() !== 6) {
                                    map.setView([latlng.lat, latlng.lng], map.getZoom(), {animate: true, duration: 0.7});
                                } else {
                                    map.panTo([latlng.lat, latlng.lng], {animate: true, duration: 0.7});
                                }
                            }

                            window.bridge.onPlaneClicked(
                                tooltip,

                                "Desconhecido",

                                latlng.lat,
                                latlng.lng
                            );

                        });

                    }

                }catch(e){}

            }

        }

        setTimeout(attachClicks,1500);

        </script>
        """

        m.get_root().html.add_child(Element(click_js))

        data = io.BytesIO()
        m.save(data, close_file=False)

        self.view.setHtml(data.getvalue().decode())
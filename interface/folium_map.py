import io
import folium
from folium import Element
from planes import get_lista

def build_map_html():
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
            popup=(
                f"<b>{p['icao']}</b><br/>"
                f"Callsign: {p['callsign']}<br/>"
                f"Lat: {p['trajectory'][0]['lat']:.4f}<br/>"
                f"Lon: {p['trajectory'][0]['lon']:.4f}"
            ),
            icon=folium.Icon(color="red", icon="plane", prefix="fa")
        ).add_to(m)

    channel_js = """
    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script>
    new QWebChannel(qt.webChannelTransport, function(channel) {
        window.bridge = channel.objects.bridge;
    });
    </script>
    """

    click_js = """
    <script>
    function attachClicks() {
        for (let i in window) {
            try {
                let obj = window[i];
                if (obj && obj._latlng) {
                    obj.on('click', function(e) {
                        let p = e.target;
                        window.bridge.onPlaneClicked(
                            p.getTooltip() ? p.getTooltip().getContent() : "N/A",
                            "Desconhecido",
                            p.getLatLng().lat,
                            p.getLatLng().lng
                        );
                    });
                }
            } catch (e) {}
        }
    }

    setTimeout(attachClicks, 1500);
    </script>
    """

    m.get_root().html.add_child(Element(channel_js))
    m.get_root().html.add_child(Element(click_js))

    data = io.BytesIO()
    m.save(data, close_file=False)

    return data.getvalue().decode()


def build_folium_map(planes=None):
    if planes is None:
        planes = get_lista()

    m = folium.Map(
        location=[-14.235, -51.925],
        zoom_start=4,
        tiles="CartoDB positron"
    )

    for plane in planes:
        popup_html = (
            f"<b>{plane['callsign']}</b><br/>"
            f"País: {plane['country']}<br/>"
            f"Lat: {plane['lat']:.4f}<br/>"
            f"Lon: {plane['lon']:.4f}"
        )
        folium.Marker(
            location=[plane["lat"], plane["lon"]],
            tooltip=plane["callsign"],
            popup=popup_html,
            icon=folium.Icon(color="red", icon="plane", prefix="fa")
        ).add_to(m)

    return m


def run_streamlit_app():
    try:
        import streamlit as st
        from streamlit_folium import st_folium
    except ImportError as exc:
        raise SystemExit(
            "Instale streamlit e streamlit-folium para usar esta visualização: "
            "pip install streamlit streamlit-folium"
        ) from exc

    st.set_page_config(page_title="Aviões em tempo real", layout="wide")
    st.title("Aviões em tempo real (OpenSky)")

    planes = get_lista()

    if not planes:
        st.warning("Nenhum avião encontrado no momento.")
        return

    selected_plane = st.session_state.get("selected_plane")
    if selected_plane is None:
        selected_plane = planes[0]

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Detalhes do voo")
        if selected_plane:
            st.metric("Callsign", selected_plane["callsign"])
            st.write(f"País: {selected_plane['country']}")
            st.write(f"Latitude: {selected_plane['lat']:.4f}")
            st.write(f"Longitude: {selected_plane['lon']:.4f}")
        else:
            st.info("Clique em um marcador para ver os detalhes.")

    with right_col:
        map_obj = build_folium_map(planes)
        try:
            map_state = st_folium(
                map_obj,
                width=900,
                height=600,
                key="plane-map",
                returned_objects=["last_object_clicked", "bounds"],
            )
        except TypeError:
            map_state = st_folium(map_obj, width=900, height=600, key="plane-map")

        clicked = map_state.get("last_object_clicked") or {}
        tooltip = clicked.get("tooltip")

        if tooltip:
            match = next((plane for plane in planes if plane["callsign"] == tooltip), None)
            if match:
                st.session_state["selected_plane"] = match

        if selected_plane and selected_plane not in planes:
            st.session_state["selected_plane"] = planes[0]
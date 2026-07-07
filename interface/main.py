import sys


def main():
    if "--streamlit" in sys.argv:
        try:
            from folium_map import run_streamlit_app

            run_streamlit_app()
            return
        except Exception as exc:
            print(f"Unable to start Streamlit mode: {exc}")
            sys.exit(1)

    try:
        from PyQt6.QtWidgets import QApplication
        from map_window import MapWindow
    except Exception as exc:
        print(f"Unable to start the desktop app: {exc}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
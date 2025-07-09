from threading import Thread
from kivy.app import App
from kivy.uix.label import Label
import webbrowser
import app as backend_app

def run_flask():
    backend_app.app.run(host='0.0.0.0', port=5000)

class FaceApp(App):
    def build(self):
        Thread(target=run_flask, daemon=True).start()
        webbrowser.open("http://localhost:5000")
        return Label(text="FaceApp Running...")

if __name__ == "__main__":
    FaceApp().run()

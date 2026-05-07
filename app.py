import importlib.util
from pathlib import Path


APP_FILE = Path(__file__).resolve().parent / "app" / "app.py"
spec = importlib.util.spec_from_file_location("gradio_app", APP_FILE)
gradio_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradio_app)
demo = gradio_app.demo


if __name__ == "__main__":
    gradio_app.launch_app()

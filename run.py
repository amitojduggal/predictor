import sys
import types

# 1. Inject the missing imghdr module into the global system modules
# This must happen before we import anything from streamlit
imghdr = types.ModuleType("imghdr")
imghdr.what = lambda file, h=None: None
sys.modules["imghdr"] = imghdr

# 2. Now import and run the Streamlit CLI
try:
    from streamlit.web.cli import main
except ImportError:
    # Older versions of streamlit
    from streamlit.cli import main

if __name__ == "__main__":
    # Simulate the command line: streamlit run src/app.py
    sys.argv = [
        "streamlit",
        "run",
        "src/app.py",
    ]
    main()

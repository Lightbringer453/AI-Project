import sys
import subprocess
from pathlib import Path

def main():
    base_path = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
    interface_path = base_path / "app" / "interface.py"
    
    if not interface_path.exists():
        sys.exit(1)
    
    streamlit_cmd = ["streamlit", "run", str(interface_path)]
    
    try:
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess.run(streamlit_cmd, cwd=str(base_path), startupinfo=startupinfo)
        else:
            subprocess.run(streamlit_cmd, cwd=str(base_path))
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()

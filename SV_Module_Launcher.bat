@echo off
cd "C:\Users\admin\Documents\Spark_Vertical_tools"
REM Change to your virtual environment directory
SET VENV_PATH="C:\Users\admin\Documents\venv_311\"

REM Activate the virtual environment
CALL %VENV_PATH%\Scripts\activate.bat

REM Run your Python script
python SV_Module_Launcher.py

REM Optional: deactivate the virtual environment
CALL deactivate
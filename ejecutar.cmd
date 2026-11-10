@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo No se encuentra el entorno Python del proyecto.
    echo Consulta la seccion Ejecucion del archivo README.md.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0oscilador_armonico.py" %*
set "resultado=%errorlevel%"
if not "%resultado%"=="0" echo El programa termino con un error. Revisa el mensaje anterior.
if "%~1"=="" pause
exit /b %resultado%

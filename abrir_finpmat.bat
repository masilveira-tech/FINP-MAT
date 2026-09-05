@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Criando o ambiente do FINP MAT...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :erro
)

echo Instalando ou atualizando os componentes...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :erro

echo Abrindo o FINP MAT...
".venv\Scripts\python.exe" -m streamlit run app.py
goto :fim

:erro
echo.
echo Nao foi possivel iniciar. Verifique se o Python esta instalado.
pause

:fim
endlocal

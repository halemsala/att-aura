@echo off
cd /d C:\aura
echo [FASE1] Instalando sqlite-vec...
python -m pip install --upgrade sqlite-vec
where ollama >nul 2>nul
if %errorlevel%==0 (
    echo [FASE1] Baixando modelo de embedding local...
    ollama pull nomic-embed-text
) else (
    echo [FASE1] AVISO: Ollama nao encontrado no PATH. Baixe nomic-embed-text manualmente
    echo        ou configure provider_order=["lmstudio"] em config\aura_rag.json
)
echo [FASE1] Concluido.
pause

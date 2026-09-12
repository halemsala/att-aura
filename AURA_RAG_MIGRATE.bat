@echo off
cd /d C:\aura
echo [FASE1] Migracao do historico para o RAG — DRY-RUN primeiro:
python scripts\aura_rag_migrate_history.py --dry-run
echo.
set /p CONF=Executar migracao REAL agora? (S/N): 
if /i not "%CONF%"=="S" (echo [FASE1] Abortado pelo operador. & exit /b 0)
python scripts\aura_rag_migrate_history.py
echo.
python -c "import sys; sys.path.insert(0,'.'); from agents.vector_memory import get_vector_memory; print(get_vector_memory().stats())"
pause

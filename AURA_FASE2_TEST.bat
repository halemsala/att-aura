@echo off
cd /d C:\aura
echo [FASE2] Compilacao + testes (obs + backtest)...
python -m compileall -q engine\aura_observability.py agents\rag_backtest.py
if errorlevel 1 (echo [FASE2] FALHA compilacao & exit /b 1)
python -m unittest tests.test_observability tests.test_rag_backtest -v
if errorlevel 1 (echo [FASE2] FALHA nos testes & exit /b 1)
echo [FASE2] APROVADO.
pause

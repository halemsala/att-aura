@echo off
cd /d C:\aura
echo [FASE1] Compilacao dos modulos novos...
python -m compileall -q agents\vector_memory.py agents\rag_analog.py engine\rag_routes.py engine\sse_state.py bridge\telegram_alerts.py scripts\aura_telegram_watchdog.py scripts\aura_rag_migrate_history.py
if errorlevel 1 (echo [FASE1] FALHA na compilacao & exit /b 1)
echo [FASE1] Suite de testes (offline, nao precisa de Ollama)...
python -m unittest tests.test_vector_memory tests.test_rag_analog tests.test_telegram_alerts tests.test_telegram_watchdog tests.test_rag_migrate -v
if errorlevel 1 (echo [FASE1] FALHA nos testes & exit /b 1)
echo [FASE1] APROVADO.
pause

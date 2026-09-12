@echo off
cd /d C:\aura
echo === Verificacao de arquivos da FASE 1 ===
set FALTANDO=0
for %%f in (
 "agents\vector_memory.py"
 "agents\rag_analog.py"
 "engine\rag_routes.py"
 "engine\sse_state.py"
 "desktop\ui\matriz_v22\assets\aura-stream.js"
 "bridge\telegram_alerts.py"
 "scripts\aura_telegram_watchdog.py"
 "scripts\aura_rag_migrate_history.py"
 "tests\test_vector_memory.py"
 "tests\test_rag_analog.py"
 "tests\test_telegram_alerts.py"
 "tests\test_telegram_watchdog.py"
 "tests\test_rag_migrate.py"
 "skills\aura-rag-analog\SKILL.md"
 "config\aura_rag.json"
 "config\aura_telegram.json"
 "config\aura_watchdog.json"
 "AURA_FASE1_INSTALL.bat"
 "AURA_FASE1_TEST.bat"
 "AURA_RAG_MIGRATE.bat"
 "AURA_TELEGRAM_WATCHDOG.bat"
) do (
 if exist %%f (echo   OK    %%f) else (echo   FALTA %%f & set FALTANDO=1)
)
if "%FALTANDO%"=="1" (echo. & echo *** Ha arquivos faltando. *** ) else (echo. & echo *** Todos os arquivos da Fase 1 presentes. ***)
pause

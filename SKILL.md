# Skill: aura-rag-analog

## Objetivo
Operar e diagnosticar a memoria vetorial de casos analogos (RAG local) do Aura.

## Quando usar
- Antes de rodas longas de analise: checar /api/rag/health
- Quando o contexto analogico vier vazio ou "insufficient_evidence"
- Para responder "em casos parecidos, qual foi a taxa historica?"

## Comandos
- Health:  curl http://127.0.0.1:8765/api/rag/health
- Stats:   curl http://127.0.0.1:8765/api/rag/stats
- Busca manual: POST /api/rag/search {"league":"...","minute":70,"line":1.0}
- Migrar historico: AURA_RAG_MIGRATE.bat (dry-run antes, sempre)
- Watchdog: AURA_TELEGRAM_WATCHDOG.bat

## Como ler o resultado
- status=ok: analogous.over_rate = taxa over historica (excl. push);
  confidence sobe com n>=10 resolvidos e similaridade media alta.
- status=insufficient_evidence: NAO e erro. Base pequena ou poucos analogos.
  Prosseguir sem contexto analogico.
- status=error: provider caiu. Analise continua sem RAG (fail-safe por design).

## Regras de seguranca
1. RAG e evidencia advisory: nunca publicar recomendacao baseada apenas nele.
2. Campos sensiveis (token/cookie/senha/api_key) sao bloqueados antes do embed.
3. Nao editar o banco vetorial manualmente; usar migrador ou indexacao automatica.
4. Migracao e idempotente: reexecutar nao duplica (log por PK + hash de conteudo).

## Diagnostico rapido
1. /api/rag/health: enabled=true e vec_available=true?
2. total_cases=0? Rodar AURA_RAG_MIGRATE.bat.
3. provider null? Ollama (11434) ou LM Studio (1234) ligado?
4. Sempre insufficient_evidence? Normal com base pequena (min_cases=5).

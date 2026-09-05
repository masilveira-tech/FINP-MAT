# FINP MAT

**Organizador financeiro local com leitura de extratos, revisão assistida e IA privada.**

O FINP MAT transforma extratos em lançamentos revisáveis, preserva o histórico em SQLite e usa o agente local **MATHEWZINHO** para explicar padrões de gastos e aprender regras de categorização. Foi desenvolvido como um projeto de portfólio para demonstrar a aplicação de Python, análise de dados, OCR, persistência e IA local a uma rotina financeira real.

> Privacidade por padrão: este repositório não contém extratos, bancos de dados, backups, e-books nem senhas reais.

## O que o projeto demonstra

- **Importação e OCR:** PDFs, CSV e XLSX; leitura nativa quando disponível e OCR com PyMuPDF + Tesseract para documentos escaneados.
- **Tratamento de dados imperfeitos:** normalização de datas, valores e estabelecimentos; identificação de entrada, saída, Pix, recorrências, parcelas e incertezas de leitura.
- **Histórico persistente:** SQLite, deduplicação por hash/fingerprint, auditoria de alterações e pontos de restauração.
- **Revisão visual:** edição de lançamentos, divisão de despesas, correção em lote por vários estabelecimentos e aprendizado das correções.
- **Análise financeira:** fluxo de caixa, categorias, orçamentos, metas, contas, faturas, recorrências, anomalias e relatórios.
- **MATHEWZINHO:** RAG com PDFs de educação financeira, cálculos controlados sobre o histórico e integração opcional com Ollama local.

## Arquitetura

```text
PDF / CSV / XLSX
       ↓
Extração + OCR + normalização
       ↓
Revisão humana e regras aprendidas
       ↓
SQLite auditável
       ↓
Dashboard + MATHEWZINHO + exportações
```

## Tecnologias

Python · Streamlit · Pandas · SQLite · Plotly · PyMuPDF · Tesseract OCR · pdfplumber · Ollama · Cryptography

## Executar localmente

### Windows — forma simples

Após extrair o projeto, execute `abrir_finpmat.bat`.

### Terminal

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Abra `http://localhost:8501` no navegador.

### OCR para PDFs escaneados

Instale o [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) e abra um novo PowerShell. O FINP MAT localiza automaticamente as pastas comuns de instalação do Windows.

### MATHEWZINHO com IA local

Instale o [Ollama](https://ollama.com/download/windows) e execute:

```powershell
ollama pull llama3.2:3b
ollama serve
```

Mantenha o Ollama em execução e abra o FINP MAT em outro terminal. Sem Ollama, a consulta documental ainda funciona, mas sem geração de linguagem natural pelo modelo.

## Demonstração segura

No primeiro acesso, crie um usuário local. Em **Configurações**, use **Carregar dados de demonstração** para explorar as telas sem importar informações pessoais.

## Qualidade

O repositório inclui testes para regras de direção Pix, aprendizado de categorias, variações de OCR, deduplicação, divisão de lançamentos e restauração segura.

```powershell
python -m unittest discover -s tests -v
```

## Limites e uso responsável

O FINP MAT é uma ferramenta educacional e de organização pessoal; não constitui recomendação de investimento, crédito ou planejamento financeiro individual. Sempre revise leituras de OCR antes de tomar decisões com base nos gráficos.

## Autor

**Matheus Silveira** · Ciências Econômicas — UFF

Veja também o [texto de apresentação e roteiro de vídeo](docs/PORTFOLIO_LINKEDIN.md) e o [guia para publicação no GitHub](docs/COMO_PUBLICAR_NO_GITHUB.md).

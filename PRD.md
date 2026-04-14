# PRD.md — Project Viability Classifier (Web)

## 1. Visão Geral

**Nome do Produto:** Project Viability Classifier  
**Versão:** 1.0  
**Data:** 2026-04-14  
**Autor:** Leandro  

### Problema
O modelo de classificação de viabilidade de projetos existe hoje como um script Python isolado, sem interface, sem histórico e sem forma prática de uso por pessoas não técnicas.

### Solução
Uma aplicação web que permite ao usuário fazer upload de dados de projetos, treinar o modelo de regressão logística e obter previsões de viabilidade com visualização clara dos resultados.

---

## 2. Objetivos

| # | Objetivo | Critério de Sucesso |
|---|----------|---------------------|
| 1 | Permitir upload de CSV para treino do modelo | Upload funciona e modelo é treinado sem erro |
| 2 | Receber novos projetos via formulário web | Formulário valida e envia dados corretamente |
| 3 | Exibir previsão de viabilidade com probabilidade | Resultado exibido com % e classificação visual |
| 4 | Exibir métricas do modelo treinado | Precision, Recall e F1 visíveis na interface |

---

## 3. Usuários-Alvo

- **Analistas de projetos** que precisam avaliar viabilidade rapidamente
- **Gestores** que querem visão consolidada sem abrir código

---

## 4. Funcionalidades (Escopo v1.0)

### 4.1 Upload de CSV para Treino
- O usuário faz upload de um arquivo `.csv`
- O backend valida as colunas obrigatórias: `investment`, `expected_return`, `impact_score`, `viability`
- O modelo é treinado e salvo em disco
- A interface confirma o treinamento e exibe as métricas

### 4.2 Formulário de Previsão
- Campos: `investment`, `expected_return`, `impact_score`
- Validação client-side (campos obrigatórios, valores numéricos positivos)
- Envio via fetch para a API

### 4.3 Resultado de Viabilidade
- Card de resultado com:
  - Classificação: **Viável** / **Inviável**
  - Probabilidade em % com barra de progresso
  - Cor semântica: verde (viável) / vermelho (inviável)

### 4.4 Painel de Métricas do Modelo
- Exibe após treino ou ao carregar (se modelo já existir):
  - Accuracy, Precision, Recall, F1-Score (classes 0 e 1)
  - Status do modelo: "Treinado" ou "Aguardando CSV"

---

## 5. Fora do Escopo (v1.0)

- Autenticação de usuários
- Histórico de previsões
- Exportação de resultados
- Múltiplos modelos simultâneos

---

## 6. Requisitos Não-Funcionais

| Requisito | Detalhe |
|-----------|---------|
| Performance | Resposta da API em menos de 2s para previsão |
| Compatibilidade | Chrome, Firefox, Safari (últimas 2 versões) |
| Responsividade | Layout funcional em desktop e tablet |
| Persistência | Modelo salvo em disco via joblib entre sessões |

---

## 7. Fluxo Principal

```
Usuário abre a app
    │
    ├─► Modelo existe? ──► SIM ──► Exibe métricas + formulário de previsão
    │
    └─► NÃO ──► Exibe área de upload de CSV
                    │
                    └─► Upload feito ──► Treina modelo ──► Exibe métricas + formulário
                                                                │
                                                        Usuário preenche projeto
                                                                │
                                                        Exibe resultado (viável/inviável + %)
```

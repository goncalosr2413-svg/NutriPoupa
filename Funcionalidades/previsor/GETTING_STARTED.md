# 🥑 NutriPoupa - Sistema de Alerta de Inflação
## Guia de Início Rápido

---

## 📦 O que foi criado?

Sistema completo de **Time Series Forecasting** para prever preços e alertar utilizadores sobre inflação.

### ✅ Ficheiros Entregues

| Ficheiro | Descrição |
|----------|-----------|
| `price_forecasting.py` | **Módulo principal** - Classe `PriceForecaster` com 3 modelos (Prophet, ARIMA, Linear) |
| `main_pipeline.py` | **Pipeline completo** - Treino, previsões e scan de alertas |
| `demo_rapido.py` | **Demo executável** - Teste rápido do sistema (< 1 min) |
| `config.py` | **Configurações** - Thresholds, modelos, notificações, etc. |

---

## 🚀 Como Usar?

### Opção 1: Demo Rápido (Recomendado para começar)

```bash
# Instalar dependências mínimas
pip install pandas numpy scikit-learn

# Executar demo
python 3-demo_rapido.py
```

**Output esperado:**
```
🚨 ALERTA DETECTADO!
   ⚠️ Alerta: Preço pode subir 10.2% nos próximos 30 dias!
   Produto: PROD_FRANGO_001
   💡 Recomendamos fazer stock-up deste produto agora.
```

---

### Opção 2: Pipeline Completo

```bash
# Executar pipeline com seus dados
python main_pipeline.py
```

Este script:
1. ✅ Carrega dados históricos (CSV ou gera exemplos)
2. ✅ Treina modelos para todos os produtos
3. ✅ Faz previsões para T+30 dias
4. ✅ Escaneia e dispara alertas
5. ✅ Exporta alertas para CSV

---

### Opção 3: Uso Programático

```python
from price_forecasting import PriceForecaster
import pandas as pd

# 1. Carregar seus dados
df = pd.read_csv('seus_dados.csv')
# Formato esperado: Data, ProductID, Categoria, PreçoMédio

# 2. Criar e treinar modelo
forecaster = PriceForecaster(model_type='linear')  # ou 'prophet'
metrics = forecaster.train(df, product_id='PROD_001')

print(f"MAE: {metrics['mae']:.2f}")

# 3. Fazer previsão
prediction = forecaster.predict('PROD_001', days_ahead=30)
print(f"Preço previsto: €{prediction['predicted_price']:.2f}")

# 4. Verificar alerta (LÓGICA PRINCIPAL)
alert = forecaster.check_alert_trigger(
    product_id='PROD_001',
    days_ahead=30,
    threshold_increase=0.05,  # 5% aumento
    min_confidence=0.85       # 85% confiança
)

if alert:
    print(f"🚨 {alert['message']}")
    print(f"💡 {alert['recommendation']}")
```

---

## 🎯 Lógica do Alerta (Requisito Técnico)

```python
# Critérios de Disparo
IF Preço_Previsto > Preço_Atual * 1.05  # Aumento > 5%
   AND Confiança_Modelo >= 0.85          # Confiança >= 85%
THEN
   Disparar_Alerta(
       produto=produto,
       aumento_percentual=X%,
       recomendacao="Stock-up agora!"
   )
```

### Código da Função de Verificação

```python
def check_alert_trigger(
    self, 
    product_id: str, 
    days_ahead: int = 30,
    threshold_increase: float = 0.05,  # 5%
    min_confidence: float = 0.85       # 85%
) -> Optional[Dict]:
    """
    Verifica se deve disparar alerta
    
    Returns:
        Dict com alerta OU None
    """
    # 1. Obter previsão
    prediction = self.predict(product_id, days_ahead)
    current_price = self.models[product_id]['last_price']
    
    # 2. Calcular aumento percentual
    price_increase = (prediction['predicted_price'] - current_price) / current_price
    
    # 3. CONDIÇÕES DO GATILHO
    should_alert = (
        price_increase > threshold_increase and
        prediction['confidence'] >= min_confidence
    )
    
    # 4. Retornar alerta se condições satisfeitas
    if should_alert:
        return {
            'alert_type': 'PRICE_INCREASE_WARNING',
            'product_id': product_id,
            'current_price': current_price,
            'predicted_price': prediction['predicted_price'],
            'price_increase_percent': price_increase * 100,
            'confidence': prediction['confidence'] * 100,
            'message': f"⚠️ Preço pode subir {price_increase*100:.1f}%!",
            'recommendation': "Recomendamos fazer stock-up deste produto agora."
        }
    
    return None
```

---

## 🤖 Modelos Disponíveis

### 1. **Linear Regression** (Default - sempre funciona)
- ✅ Sem dependências pesadas
- ✅ Rápido de treinar
- ⚠️ Precisão limitada
- **Usar para:** MVP, testes iniciais

### 2. **Prophet** (Recomendado para produção)
- ✅ Melhor precisão
- ✅ Detecta sazonalidade
- ✅ Intervalos de confiança calibrados
- ⚠️ Requer instalação: `pip install prophet`
- **Usar para:** Produção

### 3. **ARIMA** (Clássico)
- ✅ Bem estabelecido
- ✅ Bom para curto prazo
- ⚠️ Mais lento
- ⚠️ Requer instalação: `pip install statsmodels`
- **Usar para:** Benchmarking

---

## 📊 Formato de Dados Esperado

```csv
Data,ProductID,Categoria,PreçoMédio
2024-01-01,PROD_001,Laticínios,1.25
2024-01-02,PROD_001,Laticínios,1.27
2024-01-03,PROD_001,Laticínios,1.26
2024-01-01,PROD_002,Carne,5.40
```

**Requisitos:**
- ✅ Mínimo 30 dias de histórico por produto
- ✅ Recomendado: 90+ dias
- ✅ Ideal: 1+ ano para sazonalidade

---

## ⚙️ Configurações Principais

Edite `config.py` para personalizar:

```python
ALERT_CONFIG = {
    'forecast_horizon_days': 30,        # Prever 30 dias
    'price_increase_threshold': 0.05,   # 5% aumento
    'min_confidence': 0.85,             # 85% confiança
    
    # Categorias prioritárias (threshold menor)
    'priority_categories': ['Laticínios', 'Carne', 'Peixe'],
    'priority_threshold': 0.03,  # 3% para categorias prioritárias
}

MODEL_CONFIG = {
    'model_type': 'linear',  # Trocar para 'prophet' em produção
}
```

---

## 📈 Métricas de Performance

O sistema calcula automaticamente:

- **MAE** (Mean Absolute Error): Erro médio absoluto
- **RMSE** (Root Mean Squared Error): Raiz do erro quadrático médio
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio
- **R²** (Coeficiente de Determinação): Qualidade do fit (apenas Linear)

**Exemplo de output:**
```
✅ PROD_001 | MAE: 0.12 | RMSE: 0.15 | R²: 0.89
```

---

## 🔄 Deployment em Produção

### Cronjob Diário

```bash
# Executar todos os dias às 6h da manhã
crontab -e

# Adicionar:
0 6 * * * cd /path/to/project && python main_pipeline.py >> logs/alerts.log 2>&1
```

### Com Schedule (Python)

```python
import schedule
import time

def job():
    # Executar pipeline
    os.system('python main_pipeline.py')

schedule.every().day.at("06:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 🧪 Próximos Passos

### Imediato
1. ✅ Executar `demo_rapido.py` para ver funcionamento
2. ✅ Testar com 1-2 produtos dos seus dados reais
3. ✅ Validar thresholds (5% e 85% são bons defaults)

### Curto Prazo (1-2 semanas)
1. 🔲 Instalar Prophet: `pip install prophet`
2. 🔲 Retreinar com modelo Prophet
3. 🔲 Comparar performance Prophet vs Linear
4. 🔲 Configurar cronjob

### Médio Prazo (1 mês)
1. 🔲 Integrar com sistema de notificações (email/push)
2. 🔲 Dashboard de monitorização (Streamlit/Plotly)
3. 🔲 A/B testing de thresholds
4. 🔲 Feedback loop de utilizadores

### Longo Prazo (3+ meses)
1. 🔲 Deep Learning (LSTM/Transformer)
2. 🔲 Multi-model ensemble
3. 🔲 Auto-tuning de hiperparâmetros
4. 🔲 Integração com stock management

---

## 📚 Documentação Adicional

- **README.md**: Documentação completa do projeto
- **price_forecasting.py**: Docstrings detalhados de cada função
- **test_price_forecasting.py**: Exemplos de uso através de testes

---

## 🆘 Troubleshooting

### Erro: "Prophet não disponível"
```bash
pip install prophet
# macOS: brew install cmake (se necessário)
```

### Erro: "Produto tem apenas X registos. Mínimo: 30"
- Solução: Aguardar mais dados ou reduzir `min_data_points` em `config.py`

### Erro: "Model for product X not trained"
- Solução: Executar `forecaster.train(df, product_id)` primeiro

### Performance baixa (R² < 0.5)
- Trocar para Prophet
- Aumentar histórico de dados
- Verificar qualidade dos dados (outliers, missing values)

---

## 🎓 Conceitos Técnicos

### Time Series Forecasting
Prever valores futuros baseado em padrões históricos

### Prophet
Modelo aditivo que decompõe série em: tendência + sazonalidade + feriados + erro

### ARIMA
AutoRegressive Integrated Moving Average - modelo clássico para séries estacionárias

### Confidence Interval
Intervalo onde o valor real provavelmente estará (85% de confiança = 85% de probabilidade)

---

## ✅ Checklist de Validação

Antes de colocar em produção, verificar:

- [ ] Testei com dados reais
- [ ] MAE < 10% do preço médio
- [ ] Pelo menos 30 dias de histórico por produto
- [ ] Thresholds validados com equipa de negócio
- [ ] Cronjob configurado e testado
- [ ] Logs a funcionar
- [ ] Notificações integradas
- [ ] Dashboard de monitorização implementado
- [ ] Testes unitários a passar: `pytest test_price_forecasting.py`

---

## 📞 Suporte

Dúvidas? Problemas?

1. Consultar README.md
2. Ver código de exemplo em `demo_rapido.py`
3. Executar testes: `pytest test_price_forecasting.py -v`
4. Contactar equipa de Data Science

---

**🎉 Boa sorte com o NutriPoupa! 🥑**

Sistema desenvolvido para prever inflação e ajudar utilizadores a poupar dinheiro através de stock-up inteligente.

---

_Última atualização: 2026-02-08_
_Versão: 1.0 (MVP)_

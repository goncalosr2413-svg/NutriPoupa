#!/usr/bin/env python3
"""
DEMO RÁPIDO - Sistema de Alerta de Inflação NutriPoupa
Execute este script para ver o sistema em ação em menos de 1 minuto!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("🥑 NUTRIPOUPA - DEMO RÁPIDO DO ALERTA DE INFLAÇÃO")
print("=" * 70)
print()

# =====================================================
# 1. GERAR DADOS DE EXEMPLO
# =====================================================
print("📊 Gerando dados de exemplo...")

np.random.seed(42)
dates = pd.date_range(start='2024-08-01', periods=90, freq='D')

# Produto 1: Com inflação (preço a subir)
data_inflacao = []
base_price = 3.0
for i, date in enumerate(dates):
    price = base_price * (1.005 ** i) + np.random.normal(0, 0.05)  # 0.5% por dia
    data_inflacao.append({
        'Data': date.strftime('%Y-%m-%d'),
        'ProductID': 'PROD_FRANGO_001',
        'Categoria': 'Carne',
        'PreçoMédio': round(price, 2)
    })

# Produto 2: Estável (sem inflação)
data_estavel = []
for date in dates:
    price = 1.50 + np.random.normal(0, 0.02)
    data_estavel.append({
        'Data': date.strftime('%Y-%m-%d'),
        'ProductID': 'PROD_LEITE_002',
        'Categoria': 'Laticínios',
        'PreçoMédio': round(price, 2)
    })

df = pd.DataFrame(data_inflacao + data_estavel)
print(f"✅ {len(df)} registos criados para 2 produtos")
print()

# =====================================================
# 2. IMPORTAR E TREINAR MODELO
# =====================================================
print("🤖 Treinando modelos (Linear Baseline)...")

from price_forecasting import PriceForecaster

forecaster = PriceForecaster(model_type='linear')

# Treinar produto com inflação
metrics1 = forecaster.train(df, 'PROD_FRANGO_001')
print(f"   ✅ PROD_FRANGO_001 | MAE: {metrics1['mae']:.3f} | RMSE: {metrics1['rmse']:.3f}")

# Treinar produto estável
metrics2 = forecaster.train(df, 'PROD_LEITE_002')
print(f"   ✅ PROD_LEITE_002 | MAE: {metrics2['mae']:.3f} | RMSE: {metrics2['rmse']:.3f}")
print()

# =====================================================
# 3. FAZER PREVISÕES
# =====================================================
print("🔮 Fazendo previsões para 30 dias à frente...")
print()

for product_id in ['PROD_FRANGO_001', 'PROD_LEITE_002']:
    current_price = forecaster.models[product_id]['last_price']
    prediction = forecaster.predict(product_id, days_ahead=30)
    
    change = ((prediction['predicted_price'] - current_price) / current_price) * 100
    
    print(f"📦 {product_id}")
    print(f"   Categoria: {forecaster.models[product_id]['categoria']}")
    print(f"   Preço atual: €{current_price:.2f}")
    print(f"   Preço previsto: €{prediction['predicted_price']:.2f}")
    
    if change > 0:
        print(f"   Variação: +{change:.1f}% 📈")
    else:
        print(f"   Variação: {change:.1f}% 📉")
    print()

# =====================================================
# 4. VERIFICAR ALERTAS
# =====================================================
print("⚠️  Verificando alertas de inflação...")
print("-" * 70)
print()

alert_count = 0

for product_id in ['PROD_FRANGO_001', 'PROD_LEITE_002']:
    alert = forecaster.check_alert_trigger(
        product_id=product_id,
        days_ahead=30,
        threshold_increase=0.05,  # 5%
        min_confidence=0.70       # 70% (baixo para demo)
    )
    
    if alert:
        alert_count += 1
        print(f"🚨 ALERTA DETECTADO!")
        print(f"   {alert['message']}")
        print(f"   Produto: {alert['product_id']}")
        print(f"   Categoria: {alert['categoria']}")
        print(f"   Preço atual: €{alert['current_price']}")
        print(f"   Preço previsto: €{alert['predicted_price']}")
        print(f"   Aumento: +{alert['price_increase_percent']}%")
        print(f"   Confiança: {alert['confidence']}%")
        print(f"   💡 {alert['recommendation']}")
        print()
    else:
        print(f"✅ {product_id}: Sem alerta (preço estável)")
        print()

print("-" * 70)

if alert_count > 0:
    print(f"\n🎯 RESULTADO: {alert_count} alerta(s) disparado(s)!")
else:
    print("\n✅ RESULTADO: Nenhum alerta de inflação detectado")

print()
print("=" * 70)
print("✨ DEMO CONCLUÍDA COM SUCESSO!")
print("=" * 70)
print()
print("📚 Para saber mais, consulte:")
print("   • README.md - Documentação completa")
print("   • main_pipeline.py - Pipeline completo")
print("   • config.py - Configurações")
print("   • test_price_forecasting.py - Testes unitários")
print()
print("🚀 Próximos passos:")
print("   1. Instalar Prophet: pip install prophet")
print("   2. Testar com seus dados reais: main_pipeline.py")
print("   3. Configurar cronjob para execução diária")
print()

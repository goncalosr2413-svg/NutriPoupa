"""
NutriPoupa - Pipeline Principal
Exemplo completo de uso do sistema de alertas de inflação
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from price_forecasting import (
    PriceForecaster, 
    train_multiple_products, 
    scan_all_products_for_alerts
)


def generate_sample_data(num_products: int = 10, days: int = 180) -> pd.DataFrame:
    """
    Gera dados de exemplo para testar o sistema
    
    Args:
        num_products: Número de produtos a simular
        days: Número de dias de histórico
        
    Returns:
        DataFrame com dados sintéticos
    """
    np.random.seed(42)
    
    # Definir categorias e produtos
    categorias = ['Laticínios', 'Fruta', 'Carne', 'Peixe', 'Cereais', 'Bebidas']
    produtos = {
        'Laticínios': ['PROD_LEITE_001', 'PROD_QUEIJO_002'],
        'Fruta': ['PROD_BANANA_003', 'PROD_MACA_004'],
        'Carne': ['PROD_FRANGO_005', 'PROD_VACA_006'],
        'Peixe': ['PROD_SALMAO_007', 'PROD_ATUM_008'],
        'Cereais': ['PROD_ARROZ_009', 'PROD_MASSA_010'],
        'Bebidas': ['PROD_AGUA_011', 'PROD_SUMO_012']
    }
    
    # Flatten produtos
    all_products = [(cat, prod) for cat, prods in produtos.items() for prod in prods[:num_products//len(categorias)+1]]
    all_products = all_products[:num_products]
    
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    for categoria, product_id in all_products:
        # Preço base aleatório
        base_price = np.random.uniform(1.5, 15.0)
        
        # Tendência (alguns produtos com inflação, outros estáveis)
        if np.random.random() > 0.5:
            # Produto com tendência de subida (INFLAÇÃO)
            trend = np.random.uniform(0.001, 0.003)  # 0.1% a 0.3% por dia
            volatility = 0.05
        else:
            # Produto estável
            trend = np.random.uniform(-0.0005, 0.0005)
            volatility = 0.03
        
        # Sazonalidade semanal
        weekly_pattern = np.random.uniform(0.9, 1.1, size=7)
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            
            # Componentes do preço
            trend_component = base_price * (1 + trend) ** i
            seasonal_component = weekly_pattern[date.weekday()]
            noise = np.random.normal(1, volatility)
            
            price = trend_component * seasonal_component * noise
            
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'ProductID': product_id,
                'Categoria': categoria,
                'PreçoMédio': round(price, 2)
            })
    
    return pd.DataFrame(data)


def main():
    """
    Pipeline completo de demonstração
    """
    print("=" * 80)
    print("🥑 NUTRIPOUPA - SISTEMA DE ALERTA DE INFLAÇÃO")
    print("=" * 80)
    print()
    
    # ==========================================
    # 1. CARREGAR/GERAR DADOS
    # ==========================================
    print("📁 PASSO 1: Carregar dados históricos")
    print("-" * 80)
    
    # Opção A: Carregar CSV real
    # df = pd.read_csv('historico_precos.csv')
    
    # Opção B: Gerar dados de exemplo
    print("   Gerando dados sintéticos de exemplo...")
    df = generate_sample_data(num_products=12, days=180)
    
    print(f"   ✓ Dataset carregado: {len(df)} registos")
    print(f"   ✓ Produtos únicos: {df['ProductID'].nunique()}")
    print(f"   ✓ Período: {df['Data'].min()} até {df['Data'].max()}")
    print(f"   ✓ Categorias: {', '.join(df['Categoria'].unique())}")
    print()
    
    # Mostrar amostra
    print("   Amostra dos dados:")
    print(df.head(10).to_string(index=False))
    print()
    
    # ==========================================
    # 2. TREINAR MODELOS
    # ==========================================
    print("🤖 PASSO 2: Treinar modelos de previsão")
    print("-" * 80)
    
    # Escolher modelo (prophet > arima > linear)
    # Testar com diferentes modelos:
    
    print("\n[A] Treinando com Linear Regression (Baseline - sempre funciona)")
    forecaster_linear = train_multiple_products(df, model_type='linear')
    
    # Descomentar se tiver Prophet instalado:
    # print("\n[B] Treinando com Prophet (Recomendado)")
    # forecaster_prophet = train_multiple_products(df, model_type='prophet')
    
    # Descomentar se tiver ARIMA instalado:
    # print("\n[C] Treinando com ARIMA")
    # forecaster_arima = train_multiple_products(df, model_type='arima')
    
    print("\n✅ Treino concluído!")
    print()
    
    # ==========================================
    # 3. FAZER PREVISÕES
    # ==========================================
    print("🔮 PASSO 3: Fazer previsões individuais")
    print("-" * 80)
    
    # Exemplo: prever preço de um produto específico
    example_product = df['ProductID'].iloc[0]
    
    print(f"\n📦 Produto: {example_product}")
    prediction = forecaster_linear.predict(example_product, days_ahead=30)
    
    current_price = forecaster_linear.models[example_product]['last_price']
    
    print(f"   • Preço atual: €{current_price:.2f}")
    print(f"   • Preço previsto (30 dias): €{prediction['predicted_price']:.2f}")
    print(f"   • Intervalo de confiança: €{prediction['lower_bound']:.2f} - €{prediction['upper_bound']:.2f}")
    print(f"   • Confiança: {prediction['confidence']*100:.0f}%")
    print(f"   • Data da previsão: {prediction['prediction_date'].strftime('%Y-%m-%d')}")
    
    change_pct = ((prediction['predicted_price'] - current_price) / current_price) * 100
    if change_pct > 0:
        print(f"   • Variação esperada: +{change_pct:.2f}% 📈")
    else:
        print(f"   • Variação esperada: {change_pct:.2f}% 📉")
    print()
    
    # ==========================================
    # 4. VERIFICAR ALERTAS (LÓGICA PRINCIPAL)
    # ==========================================
    print("⚠️  PASSO 4: Escanear todos os produtos para alertas")
    print("-" * 80)
    
    alerts = scan_all_products_for_alerts(
        forecaster_linear,
        days_ahead=30,
        threshold=0.05,  # 5% aumento
        min_confidence=0.70  # Baixamos para 70% porque Linear tem menos confiança
    )
    
    print(f"\n🔍 Encontrados {len(alerts)} alertas de inflação!")
    print()
    
    if alerts:
        print("📢 ALERTAS A DISPARAR:")
        print("=" * 80)
        
        for i, alert in enumerate(alerts, 1):
            print(f"\n🚨 ALERTA #{i}")
            print(f"   Produto: {alert['product_id']}")
            print(f"   Categoria: {alert['categoria']}")
            print(f"   {alert['message']}")
            print(f"   ")
            print(f"   💰 Preço atual: €{alert['current_price']}")
            print(f"   📈 Preço previsto: €{alert['predicted_price']} (em {alert['days_ahead']} dias)")
            print(f"   📊 Aumento esperado: +{alert['price_increase_percent']}%")
            print(f"   🎯 Confiança: {alert['confidence']}%")
            print(f"   📅 Data da previsão: {alert['prediction_date']}")
            print(f"   ")
            print(f"   💡 {alert['recommendation']}")
            print("-" * 80)
    else:
        print("✅ Nenhum alerta de inflação significativa detectado.")
        print("   Todos os produtos estão com preços estáveis.")
    
    print()
    
    # ==========================================
    # 5. EXPORTAR ALERTAS (OPCIONAL)
    # ==========================================
    if alerts:
        print("💾 PASSO 5: Exportar alertas")
        print("-" * 80)
        
        alerts_df = pd.DataFrame(alerts)
        alerts_df.to_csv('alertas_inflacao.csv', index=False)
        print(f"   ✓ Alertas exportados para: alertas_inflacao.csv")
        print()
    
    # ==========================================
    # 6. MÉTRICAS DO MODELO
    # ==========================================
    print("📊 PASSO 6: Métricas dos modelos")
    print("-" * 80)
    
    print("\nPerformance por produto (Linear Baseline):")
    for product_id, metrics in list(forecaster_linear.metrics.items())[:5]:
        print(f"   {product_id}")
        print(f"      MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | R²: {metrics.get('r2', 0):.3f}")
    print()
    
    print("=" * 80)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print()
    print("🎯 PRÓXIMOS PASSOS:")
    print("   1. Integrar com base de dados de produção")
    print("   2. Instalar Prophet para melhor precisão: pip install prophet")
    print("   3. Configurar cronjob para executar diariamente")
    print("   4. Integrar alertas com sistema de notificações")
    print("   5. Adicionar dashboard de monitorização")
    print()


if __name__ == "__main__":
    main()

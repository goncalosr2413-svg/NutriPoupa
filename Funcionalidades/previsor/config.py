"""
NutriPoupa - Configuração do Sistema de Alertas
Parâmetros configuráveis para o modelo de previsão
"""

# =====================================================
# CONFIGURAÇÕES DO MODELO
# =====================================================

MODEL_CONFIG = {
    # Tipo de modelo a usar ('prophet', 'arima', 'linear')
    'model_type': 'linear',  # Trocar para 'prophet' em produção
    
    # Requisitos mínimos de dados
    'min_data_points': 30,  # Mínimo de dias de histórico
    'max_missing_days': 7,  # Máximo de dias em falta consecutivos
}

# =====================================================
# CONFIGURAÇÕES DOS ALERTAS
# =====================================================

ALERT_CONFIG = {
    # Horizonte de previsão
    'forecast_horizon_days': 30,  # Prever preços para T+30 dias
    
    # Thresholds para disparo de alerta
    'price_increase_threshold': 0.05,  # 5% de aumento
    'min_confidence': 0.85,  # 85% de confiança mínima
    
    # Categorias prioritárias (disparar alerta mesmo com threshold menor)
    'priority_categories': ['Laticínios', 'Carne', 'Peixe'],
    'priority_threshold': 0.03,  # 3% para categorias prioritárias
    
    # Limites de alertas
    'max_alerts_per_user': 10,  # Máximo de alertas por utilizador
    'cooldown_period_days': 7,  # Não alertar mesmo produto nos próximos 7 dias
}

# =====================================================
# CONFIGURAÇÕES DO PROPHET (se usado)
# =====================================================

PROPHET_CONFIG = {
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'seasonality_mode': 'multiplicative',
    'changepoint_prior_scale': 0.05,
    'interval_width': 0.85,  # 85% intervalo de confiança
}

# =====================================================
# CONFIGURAÇÕES DO ARIMA (se usado)
# =====================================================

ARIMA_CONFIG = {
    'default_order': (1, 1, 1),  # (p, d, q)
    'seasonal': False,
    'trend': 'c',  # 'c' = constante, 't' = tendência, 'ct' = ambos
}

# =====================================================
# CONFIGURAÇÕES DE NOTIFICAÇÃO
# =====================================================

NOTIFICATION_CONFIG = {
    # Canais de notificação
    'channels': ['email', 'push', 'in_app'],
    
    # Templates de mensagens
    'templates': {
        'email_subject': '🚨 Alerta NutriPoupa: {categoria} pode ficar mais caro!',
        'push_message': '⚠️ {product_name} pode subir {increase}% em {days} dias!',
        'in_app_banner': 'Stock-up recomendado: {num_products} produtos em risco de inflação',
    },
    
    # Prioridades
    'critical_increase_threshold': 0.15,  # 15% = alerta crítico
}

# =====================================================
# CONFIGURAÇÕES DE PERSISTÊNCIA
# =====================================================

STORAGE_CONFIG = {
    # Paths
    'model_storage_path': './models/',
    'data_storage_path': './data/',
    'alerts_storage_path': './alerts/',
    
    # Formato de serialização
    'model_format': 'joblib',  # ou 'pickle'
    
    # Retenção
    'model_retention_days': 30,  # Manter modelos dos últimos 30 dias
    'alert_history_days': 90,  # Histórico de alertas
}

# =====================================================
# CONFIGURAÇÕES DE EXECUÇÃO
# =====================================================

EXECUTION_CONFIG = {
    # Cronjob
    'schedule': 'daily',  # 'hourly', 'daily', 'weekly'
    'execution_time': '06:00',  # Executar às 6h da manhã
    
    # Performance
    'parallel_training': True,  # Treinar modelos em paralelo
    'max_workers': 4,  # Número de workers paralelos
    
    # Logging
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file_path': './logs/nutripoupa_alerts.log',
}

# =====================================================
# FEATURE FLAGS
# =====================================================

FEATURES = {
    'enable_auto_retrain': True,  # Re-treinar modelos automaticamente
    'enable_model_comparison': True,  # Comparar múltiplos modelos
    'enable_user_feedback': True,  # Permitir feedback em alertas
    'enable_ab_testing': False,  # A/B testing de thresholds
}

# =====================================================
# MÉTRICAS E MONITORIZAÇÃO
# =====================================================

MONITORING_CONFIG = {
    # Métricas a trackear
    'track_metrics': ['mae', 'rmse', 'mape', 'precision', 'recall'],
    
    # Thresholds de qualidade
    'max_acceptable_mape': 15.0,  # MAPE máximo aceitável: 15%
    'min_acceptable_r2': 0.6,  # R² mínimo aceitável
    
    # Alertas de sistema
    'alert_on_poor_performance': True,
    'performance_check_frequency': 'weekly',
}


# =====================================================
# VALIDAÇÃO DA CONFIGURAÇÃO
# =====================================================

def validate_config():
    """Valida se as configurações são consistentes"""
    
    assert ALERT_CONFIG['price_increase_threshold'] > 0, "Threshold deve ser positivo"
    assert 0 < ALERT_CONFIG['min_confidence'] <= 1, "Confiança deve estar entre 0 e 1"
    assert ALERT_CONFIG['forecast_horizon_days'] > 0, "Horizonte deve ser positivo"
    
    print("✅ Configuração validada com sucesso!")


if __name__ == "__main__":
    print("📋 NutriPoupa - Configurações do Sistema")
    print("=" * 60)
    print(f"Modelo: {MODEL_CONFIG['model_type']}")
    print(f"Horizonte de previsão: {ALERT_CONFIG['forecast_horizon_days']} dias")
    print(f"Threshold de alerta: {ALERT_CONFIG['price_increase_threshold']*100}%")
    print(f"Confiança mínima: {ALERT_CONFIG['min_confidence']*100}%")
    print()
    validate_config()

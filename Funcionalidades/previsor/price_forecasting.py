"""
NutriPoupa - Price Forecasting Model
Alerta de Inflação: Previsão de preços para stock-up inteligente

Modelos implementados:
- Prophet (Facebook): Robusto para sazonalidade e tendências
- ARIMA: Clássico para séries temporais
- Linear Regression: Baseline simples (MVP)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Imports condicionais (instalar conforme necessário)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet não disponível. Instalar com: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️  ARIMA não disponível. Instalar com: pip install statsmodels")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PriceForecaster:
    """
    Sistema de previsão de preços para alertas de inflação
    """
    
    def __init__(self, model_type: str = 'prophet'):
        """
        Args:
            model_type: 'prophet', 'arima', ou 'linear' (baseline)
        """
        self.model_type = model_type
        self.models = {}  # Dict[ProductID, modelo]
        self.metrics = {}  # Dict[ProductID, métricas de performance]
        
    def prepare_data(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """
        Prepara dados para um produto específico
        
        Args:
            df: DataFrame com colunas ['Data', 'ProductID', 'Categoria', 'PreçoMédio']
            product_id: ID do produto a filtrar
            
        Returns:
            DataFrame filtrado e ordenado
        """
        # Filtrar produto
        product_df = df[df['ProductID'] == product_id].copy()
        
        # Converter data
        product_df['Data'] = pd.to_datetime(product_df['Data'])
        product_df = product_df.sort_values('Data')
        
        # Remover duplicatas (manter a média se houver múltiplos registos no mesmo dia)
        product_df = product_df.groupby('Data').agg({
            'PreçoMédio': 'mean',
            'Categoria': 'first',
            'ProductID': 'first'
        }).reset_index()
        
        return product_df
    
    def train_prophet(self, df: pd.DataFrame) -> Tuple[object, Dict]:
        """
        Treina modelo Prophet
        
        Args:
            df: DataFrame com colunas ['Data', 'PreçoMédio']
            
        Returns:
            (modelo_treinado, métricas)
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet não está instalado")
        
        # Preparar dados no formato Prophet (ds, y)
        prophet_df = pd.DataFrame({
            'ds': df['Data'],
            'y': df['PreçoMédio']
        })
        
        # Configurar modelo
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Melhor para preços
            changepoint_prior_scale=0.05,  # Sensibilidade a mudanças de tendência
            interval_width=0.85  # 85% de intervalo de confiança
        )
        
        # Treinar
        model.fit(prophet_df)
        
        # Calcular métricas no conjunto de treino
        predictions = model.predict(prophet_df)
        mae = mean_absolute_error(prophet_df['y'], predictions['yhat'])
        rmse = np.sqrt(mean_squared_error(prophet_df['y'], predictions['yhat']))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': np.mean(np.abs((prophet_df['y'] - predictions['yhat']) / prophet_df['y'])) * 100
        }
        
        return model, metrics
    
    def train_arima(self, df: pd.DataFrame) -> Tuple[object, Dict]:
        """
        Treina modelo ARIMA
        
        Args:
            df: DataFrame com colunas ['Data', 'PreçoMédio']
            
        Returns:
            (modelo_treinado, métricas)
        """
        if not ARIMA_AVAILABLE:
            raise ImportError("ARIMA não está instalado")
        
        # Preparar série temporal
        ts = df.set_index('Data')['PreçoMédio']
        
        # Auto-determinar ordem ARIMA (simplificado - usa (1,1,1) como default)
        # Em produção, usar auto_arima do pmdarima
        order = (1, 1, 1)  # (p, d, q)
        
        # Treinar modelo
        model = ARIMA(ts, order=order)
        fitted_model = model.fit()
        
        # Calcular métricas
        predictions = fitted_model.fittedvalues
        mae = mean_absolute_error(ts[1:], predictions[1:])  # Skip primeiro valor
        rmse = np.sqrt(mean_squared_error(ts[1:], predictions[1:]))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
        
        return fitted_model, metrics
    
    def train_linear_baseline(self, df: pd.DataFrame) -> Tuple[object, Dict]:
        """
        Treina modelo Linear simples (baseline MVP)
        
        Args:
            df: DataFrame com colunas ['Data', 'PreçoMédio']
            
        Returns:
            (modelo_treinado, métricas)
        """
        # Criar features temporais
        df = df.copy()
        df['days_since_start'] = (df['Data'] - df['Data'].min()).dt.days
        
        X = df[['days_since_start']].values
        y = df['PreçoMédio'].values
        
        # Treinar
        model = LinearRegression()
        model.fit(X, y)
        
        # Métricas
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': model.score(X, y),
            'slope': model.coef_[0],
            'intercept': model.intercept_
        }
        
        return model, metrics
    
    def train(self, df: pd.DataFrame, product_id: str) -> Dict:
        """
        Treina modelo para um produto específico
        
        Args:
            df: DataFrame histórico completo
            product_id: ID do produto
            
        Returns:
            Métricas do modelo treinado
        """
        # Preparar dados
        product_df = self.prepare_data(df, product_id)
        
        if len(product_df) < 30:
            raise ValueError(f"Produto {product_id} tem apenas {len(product_df)} registos. Mínimo: 30")
        
        # Treinar modelo apropriado
        if self.model_type == 'prophet':
            model, metrics = self.train_prophet(product_df)
        elif self.model_type == 'arima':
            model, metrics = self.train_arima(product_df)
        elif self.model_type == 'linear':
            model, metrics = self.train_linear_baseline(product_df)
        else:
            raise ValueError(f"Modelo '{self.model_type}' não reconhecido")
        
        # Guardar modelo e dados
        self.models[product_id] = {
            'model': model,
            'last_date': product_df['Data'].max(),
            'last_price': product_df['PreçoMédio'].iloc[-1],
            'categoria': product_df['Categoria'].iloc[0],
            'training_data': product_df  # Guardar para baseline linear
        }
        self.metrics[product_id] = metrics
        
        return metrics
    
    def predict(self, product_id: str, days_ahead: int = 30) -> Dict:
        """
        Faz previsão de preço para N dias à frente
        
        Args:
            product_id: ID do produto
            days_ahead: Número de dias para prever (default: 30)
            
        Returns:
            Dict com previsão e intervalo de confiança
        """
        if product_id not in self.models:
            raise ValueError(f"Modelo para produto {product_id} não foi treinado")
        
        model_data = self.models[product_id]
        model = model_data['model']
        last_date = model_data['last_date']
        
        # Data futura
        future_date = last_date + timedelta(days=days_ahead)
        
        # Previsão dependendo do modelo
        if self.model_type == 'prophet':
            future_df = pd.DataFrame({'ds': [future_date]})
            forecast = model.predict(future_df)
            
            return {
                'predicted_price': forecast['yhat'].iloc[0],
                'lower_bound': forecast['yhat_lower'].iloc[0],
                'upper_bound': forecast['yhat_upper'].iloc[0],
                'confidence': 0.85,  # Intervalo de confiança configurado
                'prediction_date': future_date
            }
            
        elif self.model_type == 'arima':
            # Prever N passos à frente
            forecast = model.forecast(steps=days_ahead)
            predicted_price = forecast.iloc[-1]
            
            # ARIMA: calcular intervalo de confiança aproximado (±2*RMSE)
            rmse = self.metrics[product_id]['rmse']
            
            return {
                'predicted_price': predicted_price,
                'lower_bound': predicted_price - 2*rmse,
                'upper_bound': predicted_price + 2*rmse,
                'confidence': 0.85,  # Aproximado
                'prediction_date': future_date
            }
            
        elif self.model_type == 'linear':
            # Calcular dias desde início
            training_data = model_data['training_data']
            start_date = training_data['Data'].min()
            days_since_start = (future_date - start_date).days
            
            X_future = np.array([[days_since_start]])
            predicted_price = model.predict(X_future)[0]
            
            # Intervalo baseado em RMSE
            rmse = self.metrics[product_id]['rmse']
            
            return {
                'predicted_price': predicted_price,
                'lower_bound': predicted_price - 2*rmse,
                'upper_bound': predicted_price + 2*rmse,
                'confidence': 0.70,  # Modelo simples = menor confiança
                'prediction_date': future_date
            }
    
    def check_alert_trigger(
        self, 
        product_id: str, 
        days_ahead: int = 30,
        threshold_increase: float = 0.05,  # 5% aumento
        min_confidence: float = 0.85
    ) -> Optional[Dict]:
        """
        LÓGICA PRINCIPAL: Verifica se deve disparar alerta de compra
        
        Args:
            product_id: ID do produto
            days_ahead: Dias para previsão (default: 30)
            threshold_increase: Aumento mínimo para alertar (default: 5%)
            min_confidence: Confiança mínima do modelo (default: 85%)
            
        Returns:
            Dict com detalhes do alerta OU None se não houver alerta
        """
        # Obter previsão
        prediction = self.predict(product_id, days_ahead)
        
        # Preço atual
        current_price = self.models[product_id]['last_price']
        predicted_price = prediction['predicted_price']
        confidence = prediction['confidence']
        
        # Calcular aumento percentual
        price_increase = (predicted_price - current_price) / current_price
        
        # CONDIÇÕES DO GATILHO
        should_alert = (
            price_increase > threshold_increase and
            confidence >= min_confidence
        )
        
        if should_alert:
            return {
                'alert_type': 'PRICE_INCREASE_WARNING',
                'product_id': product_id,
                'categoria': self.models[product_id]['categoria'],
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'price_increase_percent': round(price_increase * 100, 2),
                'confidence': round(confidence * 100, 2),
                'prediction_date': prediction['prediction_date'].strftime('%Y-%m-%d'),
                'days_ahead': days_ahead,
                'lower_bound': round(prediction['lower_bound'], 2),
                'upper_bound': round(prediction['upper_bound'], 2),
                'message': f"⚠️ Alerta: Preço pode subir {price_increase*100:.1f}% nos próximos {days_ahead} dias!",
                'recommendation': f"Recomendamos fazer stock-up deste produto agora."
            }
        
        return None


def train_multiple_products(
    df: pd.DataFrame, 
    model_type: str = 'prophet'
) -> PriceForecaster:
    """
    Treina modelos para todos os produtos no dataset
    
    Args:
        df: DataFrame histórico completo
        model_type: Tipo de modelo ('prophet', 'arima', 'linear')
        
    Returns:
        Forecaster treinado com todos os produtos
    """
    forecaster = PriceForecaster(model_type=model_type)
    
    # Obter lista de produtos únicos
    products = df['ProductID'].unique()
    
    print(f"🚀 Treinando modelos {model_type.upper()} para {len(products)} produtos...\n")
    
    for i, product_id in enumerate(products, 1):
        try:
            metrics = forecaster.train(df, product_id)
            print(f"✅ [{i}/{len(products)}] {product_id} | MAE: {metrics['mae']:.2f} | RMSE: {metrics.get('rmse', 0):.2f}")
        except Exception as e:
            print(f"❌ [{i}/{len(products)}] {product_id} | Erro: {str(e)}")
    
    return forecaster


def scan_all_products_for_alerts(
    forecaster: PriceForecaster,
    days_ahead: int = 30,
    threshold: float = 0.05,
    min_confidence: float = 0.85
) -> List[Dict]:
    """
    Escaneia todos os produtos e retorna lista de alertas
    
    Returns:
        Lista de alertas a disparar
    """
    alerts = []
    
    for product_id in forecaster.models.keys():
        alert = forecaster.check_alert_trigger(
            product_id=product_id,
            days_ahead=days_ahead,
            threshold_increase=threshold,
            min_confidence=min_confidence
        )
        
        if alert:
            alerts.append(alert)
    
    return alerts


if __name__ == "__main__":
    # Este bloco será executado apenas se rodar diretamente este ficheiro
    print("📊 NutriPoupa - Price Forecasting System")
    print("=" * 60)
    print("\nPara usar este módulo, veja o ficheiro 'main_pipeline.py'")
    print("\nModelos disponíveis:")
    print("  ✓ Prophet" if PROPHET_AVAILABLE else "  ✗ Prophet (instalar)")
    print("  ✓ ARIMA" if ARIMA_AVAILABLE else "  ✗ ARIMA (instalar)")
    print("  ✓ Linear Baseline (sempre disponível)")

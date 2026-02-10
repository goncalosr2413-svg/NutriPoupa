# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 23:23:48 2026

@author: ASUS
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import math
from itertools import combinations, permutations
from copy import deepcopy


@dataclass
class Product:
    """Produto desejado pelo utilizador"""
    name: str
    quantity: int = 1


@dataclass
class StoreProduct:
    """Produto disponível numa loja específica"""
    product_name: str
    price: float
    store_id: str


@dataclass
class Store:
    """Supermercado com localização"""
    id: str
    name: str
    latitude: float
    longitude: float


@dataclass
class UserLocation:
    """Localização do utilizador"""
    latitude: float
    longitude: float


@dataclass
class BasketScenario:
    """Cenário de compras otimizado"""
    scenario_name: str
    total_cost: float  # Custo total (produtos + deslocação)
    product_cost: float  # Custo só dos produtos
    travel_cost: float  # Custo estimado de deslocação
    total_distance_km: float  # Distância total em km
    stores_visited: List[str]  # IDs das lojas visitadas
    shopping_plan: Dict[str, List[Tuple[str, float, int]]]  # {store_id: [(produto, preço, qtd)]}
    route: List[str]  # Ordem de visita das lojas


class BasketOptimizer:
    """
    Motor de otimização de lista de compras considerando preço e distância.
    """
    
    def __init__(
        self,
        cost_per_km: float = 0.15,  # Custo estimado por km (combustível + tempo)
        time_value_per_hour: float = 10.0,  # Valor do tempo em €/hora
        avg_speed_kmh: float = 30.0  # Velocidade média urbana
    ):
        """
        Args:
            cost_per_km: Custo de combustível por km (€)
            time_value_per_hour: Valor monetário atribuído ao tempo (€/h)
            avg_speed_kmh: Velocidade média de deslocação
        """
        self.cost_per_km = cost_per_km
        self.time_value_per_hour = time_value_per_hour
        self.avg_speed_kmh = avg_speed_kmh
        
        # Custo logístico total por km (combustível + tempo)
        time_cost_per_km = (time_value_per_hour / avg_speed_kmh)
        self.total_cost_per_km = cost_per_km + time_cost_per_km
    
    
    def optimize(
        self,
        desired_products: List[Product],
        user_location: UserLocation,
        available_products: List[StoreProduct],
        stores: List[Store]
    ) -> Dict[str, BasketScenario]:
        """
        Gera 3 cenários otimizados de lista de compras.
        
        Returns:
            Dict com 3 cenários: 'global_minimum', 'best_value', 'convenience'
        """
        # Criar índices para acesso rápido
        store_dict = {s.id: s for s in stores}
        products_by_store = self._index_products_by_store(available_products)
        
        # ETAPA 1: Identificar preço mínimo global por produto
        min_prices = self._get_minimum_prices(desired_products, available_products)
        
        # ETAPA 2: Gerar cenários
        scenarios = {}
        
        # Cenário 1: GLOBAL MINIMUM (preço absoluto mínimo)
        scenarios['global_minimum'] = self._generate_global_minimum(
            desired_products, min_prices, products_by_store, 
            store_dict, user_location
        )
        
        # Cenário 2: BEST VALUE (máximo 2 lojas, melhor custo total)
        scenarios['best_value'] = self._generate_best_value(
            desired_products, products_by_store, store_dict, 
            user_location, max_stores=2
        )
        
        # Cenário 3: CONVENIENCE (tudo numa só loja)
        scenarios['convenience'] = self._generate_convenience(
            desired_products, products_by_store, store_dict, user_location
        )
        
        return scenarios
    
    
    def _index_products_by_store(
        self, 
        available_products: List[StoreProduct]
    ) -> Dict[str, Dict[str, List[StoreProduct]]]:
        """
        Organiza produtos por loja e nome.
        Returns: {store_id: {product_name: [StoreProduct]}}
        """
        index = {}
        for sp in available_products:
            if sp.store_id not in index:
                index[sp.store_id] = {}
            if sp.product_name not in index[sp.store_id]:
                index[sp.store_id][sp.product_name] = []
            index[sp.store_id][sp.product_name].append(sp)
        return index
    
    
    def _get_minimum_prices(
        self,
        desired_products: List[Product],
        available_products: List[StoreProduct]
    ) -> Dict[str, Tuple[float, str]]:
        """
        Identifica o preço mínimo global por produto.
        Returns: {product_name: (min_price, store_id)}
        """
        min_prices = {}
        for product in desired_products:
            min_price = float('inf')
            best_store = None
            
            for sp in available_products:
                if sp.product_name == product.name and sp.price < min_price:
                    min_price = sp.price
                    best_store = sp.store_id
            
            if best_store:
                min_prices[product.name] = (min_price, best_store)
        
        return min_prices
    
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calcula distância em km entre duas coordenadas usando Haversine.
        """
        R = 6371  # Raio da Terra em km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    
    def _calculate_route_distance(
        self,
        user_location: UserLocation,
        store_ids: List[str],
        store_dict: Dict[str, Store]
    ) -> Tuple[float, List[str]]:
        """
        Calcula a distância total da rota ótima (TSP simplificado).
        Usa nearest neighbor heuristic para ordem de visita.
        
        Returns: (total_distance_km, ordered_route)
        """
        if not store_ids:
            return 0.0, []
        
        if len(store_ids) == 1:
            store = store_dict[store_ids[0]]
            dist = self._haversine_distance(
                user_location.latitude, user_location.longitude,
                store.latitude, store.longitude
            )
            return dist * 2, [store_ids[0]]  # Ida e volta
        
        # Nearest Neighbor Heuristic para TSP
        unvisited = set(store_ids)
        route = []
        current_lat = user_location.latitude
        current_lon = user_location.longitude
        total_distance = 0.0
        
        while unvisited:
            nearest_store_id = None
            min_dist = float('inf')
            
            for store_id in unvisited:
                store = store_dict[store_id]
                dist = self._haversine_distance(
                    current_lat, current_lon,
                    store.latitude, store.longitude
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_store_id = store_id
            
            route.append(nearest_store_id)
            unvisited.remove(nearest_store_id)
            total_distance += min_dist
            
            store = store_dict[nearest_store_id]
            current_lat = store.latitude
            current_lon = store.longitude
        
        # Retorno à origem
        total_distance += self._haversine_distance(
            current_lat, current_lon,
            user_location.latitude, user_location.longitude
        )
        
        return total_distance, route
    
    
    def _generate_global_minimum(
        self,
        desired_products: List[Product],
        min_prices: Dict[str, Tuple[float, str]],
        products_by_store: Dict[str, Dict[str, List[StoreProduct]]],
        store_dict: Dict[str, Store],
        user_location: UserLocation
    ) -> BasketScenario:
        """
        Cenário 1: Preço absoluto mínimo (pode envolver várias lojas).
        """
        shopping_plan = {}
        product_cost = 0.0
        stores_needed = set()
        
        for product in desired_products:
            if product.name not in min_prices:
                continue
            
            min_price, store_id = min_prices[product.name]
            stores_needed.add(store_id)
            
            if store_id not in shopping_plan:
                shopping_plan[store_id] = []
            
            shopping_plan[store_id].append((product.name, min_price, product.quantity))
            product_cost += min_price * product.quantity
        
        # Calcular rota e custo de deslocação
        total_distance, route = self._calculate_route_distance(
            user_location, list(stores_needed), store_dict
        )
        travel_cost = total_distance * self.total_cost_per_km
        
        return BasketScenario(
            scenario_name="Global Minimum (Preço Absoluto Mínimo)",
            total_cost=product_cost + travel_cost,
            product_cost=product_cost,
            travel_cost=travel_cost,
            total_distance_km=total_distance,
            stores_visited=[store_dict[sid].name for sid in stores_needed],
            shopping_plan=shopping_plan,
            route=route
        )
    
    
    def _generate_best_value(
        self,
        desired_products: List[Product],
        products_by_store: Dict[str, Dict[str, List[StoreProduct]]],
        store_dict: Dict[str, Store],
        user_location: UserLocation,
        max_stores: int = 2
    ) -> BasketScenario:
        """
        Cenário 2: Melhor valor global (preço + distância), máximo 2 lojas.
        """
        best_scenario = None
        min_total_cost = float('inf')
        
        # Avaliar todas as combinações de até max_stores lojas
        store_ids = list(products_by_store.keys())
        
        for r in range(1, min(max_stores + 1, len(store_ids) + 1)):
            for store_combo in combinations(store_ids, r):
                # Verificar se esta combinação tem todos os produtos
                available_products_in_combo = set()
                for sid in store_combo:
                    available_products_in_combo.update(products_by_store[sid].keys())
                
                missing = [p.name for p in desired_products 
                          if p.name not in available_products_in_combo]
                if missing:
                    continue  # Esta combinação não serve
                
                # Otimizar compra nesta combinação de lojas
                scenario = self._optimize_for_store_combination(
                    desired_products, store_combo, products_by_store,
                    store_dict, user_location
                )
                
                if scenario and scenario.total_cost < min_total_cost:
                    min_total_cost = scenario.total_cost
                    best_scenario = scenario
        
        if best_scenario:
            best_scenario.scenario_name = "Best Value (Melhor Custo Total, Máx. 2 Lojas)"
            return best_scenario
        
        # Fallback se não encontrar solução
        return self._generate_convenience(
            desired_products, products_by_store, store_dict, user_location
        )
    
    
    def _optimize_for_store_combination(
        self,
        desired_products: List[Product],
        store_ids: Tuple[str],
        products_by_store: Dict[str, Dict[str, List[StoreProduct]]],
        store_dict: Dict[str, Store],
        user_location: UserLocation
    ) -> BasketScenario:
        """
        Otimiza a compra para uma combinação específica de lojas.
        Para cada produto, escolhe a loja mais barata dentro da combinação.
        """
        shopping_plan = {}
        product_cost = 0.0
        
        for product in desired_products:
            min_price = float('inf')
            best_store = None
            
            for store_id in store_ids:
                if product.name in products_by_store.get(store_id, {}):
                    # Pegar o produto mais barato desta loja
                    prices = [sp.price for sp in products_by_store[store_id][product.name]]
                    price = min(prices)
                    
                    if price < min_price:
                        min_price = price
                        best_store = store_id
            
            if best_store is None:
                return None  # Produto não disponível nesta combinação
            
            if best_store not in shopping_plan:
                shopping_plan[best_store] = []
            
            shopping_plan[best_store].append((product.name, min_price, product.quantity))
            product_cost += min_price * product.quantity
        
        # Calcular rota
        total_distance, route = self._calculate_route_distance(
            user_location, list(shopping_plan.keys()), store_dict
        )
        travel_cost = total_distance * self.total_cost_per_km
        
        return BasketScenario(
            scenario_name="Optimized Combination",
            total_cost=product_cost + travel_cost,
            product_cost=product_cost,
            travel_cost=travel_cost,
            total_distance_km=total_distance,
            stores_visited=[store_dict[sid].name for sid in shopping_plan.keys()],
            shopping_plan=shopping_plan,
            route=route
        )
    
    
    def _generate_convenience(
        self,
        desired_products: List[Product],
        products_by_store: Dict[str, Dict[str, List[StoreProduct]]],
        store_dict: Dict[str, Store],
        user_location: UserLocation
    ) -> BasketScenario:
        """
        Cenário 3: Tudo numa só loja (a mais barata para o cabaz completo).
        """
        best_scenario = None
        min_total_cost = float('inf')
        
        for store_id, products in products_by_store.items():
            # Verificar se tem todos os produtos
            missing = [p.name for p in desired_products if p.name not in products]
            if missing:
                continue
            
            # Calcular custo total nesta loja
            shopping_plan = {}
            product_cost = 0.0
            
            for product in desired_products:
                prices = [sp.price for sp in products[product.name]]
                min_price = min(prices)
                
                if store_id not in shopping_plan:
                    shopping_plan[store_id] = []
                
                shopping_plan[store_id].append((product.name, min_price, product.quantity))
                product_cost += min_price * product.quantity
            
            # Calcular distância
            store = store_dict[store_id]
            distance = self._haversine_distance(
                user_location.latitude, user_location.longitude,
                store.latitude, store.longitude
            ) * 2  # Ida e volta
            
            travel_cost = distance * self.total_cost_per_km
            total_cost = product_cost + travel_cost
            
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_scenario = BasketScenario(
                    scenario_name="Convenience (Uma Só Loja)",
                    total_cost=total_cost,
                    product_cost=product_cost,
                    travel_cost=travel_cost,
                    total_distance_km=distance,
                    stores_visited=[store.name],
                    shopping_plan=shopping_plan,
                    route=[store_id]
                )
        
        return best_scenario


# ============================================================
# EXEMPLO DE USO
# ============================================================

def exemplo_uso():
    """Demonstração do BasketOptimizer"""
    
    # Lista de compras do utilizador
    desired_products = [
        Product("Leite", quantity=2),
        Product("Pão", quantity=1),
        Product("Ovos", quantity=1),
        Product("Queijo", quantity=1),
    ]
    
    # Localização do utilizador (ex: Centro de Lisboa)
    user_location = UserLocation(latitude=38.7223, longitude=-9.1393)
    
    # Supermercados disponíveis
    stores = [
        Store("continente_1", "Continente Colombo", 38.7569, -9.2036),
        Store("pingo_doce_1", "Pingo Doce Saldanha", 38.7333, -9.1450),
        Store("lidl_1", "Lidl Alcântara", 38.7072, -9.1705),
        Store("mercadona_1", "Mercadona Alameda", 38.7379, -9.1382),
    ]
    
    # Base de dados de produtos (simulada)
    available_products = [
        # Continente
        StoreProduct("Leite", 0.89, "continente_1"),
        StoreProduct("Pão", 0.45, "continente_1"),
        StoreProduct("Ovos", 1.99, "continente_1"),
        StoreProduct("Queijo", 2.49, "continente_1"),
        
        # Pingo Doce
        StoreProduct("Leite", 0.95, "pingo_doce_1"),
        StoreProduct("Pão", 0.40, "pingo_doce_1"),
        StoreProduct("Ovos", 1.89, "pingo_doce_1"),
        StoreProduct("Queijo", 2.35, "pingo_doce_1"),
        
        # Lidl
        StoreProduct("Leite", 0.79, "lidl_1"),
        StoreProduct("Pão", 0.35, "lidl_1"),
        StoreProduct("Ovos", 1.69, "lidl_1"),
        StoreProduct("Queijo", 1.99, "lidl_1"),
        
        # Mercadona
        StoreProduct("Leite", 0.85, "mercadona_1"),
        StoreProduct("Pão", 0.38, "mercadona_1"),
        StoreProduct("Ovos", 1.79, "mercadona_1"),
        StoreProduct("Queijo", 2.15, "mercadona_1"),
    ]
    
    # Criar otimizador
    optimizer = BasketOptimizer(
        cost_per_km=0.15,  # €0.15/km combustível
        time_value_per_hour=12.0,  # €12/hora valor do tempo
        avg_speed_kmh=25.0  # 25 km/h velocidade urbana
    )
    
    # Gerar cenários
    scenarios = optimizer.optimize(
        desired_products=desired_products,
        user_location=user_location,
        available_products=available_products,
        stores=stores
    )
    
    # Apresentar resultados
    print("=" * 70)
    print("NUTRIPOUPA - OTIMIZAÇÃO DE LISTA DE COMPRAS")
    print("=" * 70)
    
    for key, scenario in scenarios.items():
        print(f"\n📊 {scenario.scenario_name}")
        print(f"   Custo Total: €{scenario.total_cost:.2f}")
        print(f"   ├─ Produtos: €{scenario.product_cost:.2f}")
        print(f"   └─ Deslocação: €{scenario.travel_cost:.2f} ({scenario.total_distance_km:.1f} km)")
        print(f"   Lojas: {', '.join(scenario.stores_visited)}")
        print(f"   Rota: {' → '.join(scenario.route)}")
        print(f"\n   Plano de compras:")
        for store_id, items in scenario.shopping_plan.items():
            store_name = next(s.name for s in stores if s.id == store_id)
            print(f"   🏪 {store_name}:")
            for product_name, price, qty in items:
                print(f"      • {qty}x {product_name} @ €{price:.2f} = €{price * qty:.2f}")


if __name__ == "__main__":
    exemplo_uso()
```

## 📋 **Explicação da Lógica de Ponderação**

### **Fórmula do Custo Total:**
```
Custo Total = Custo Produtos + Custo Deslocação
Custo Deslocação = Distância (km) × [Custo Combustível + Valor Tempo]
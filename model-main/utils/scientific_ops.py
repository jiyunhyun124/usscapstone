# utils/scientific_ops.py
import numpy as np

class ScientificCleanupRecovery:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        
        # [Scientific Constants] 논문/문헌 기반 파라미터
        self.k_deg_natural = 0.05   # 자연 분해율
        self.k_deg_active = 0.2     # 방제 작업 시 분해율 (4배 가속)
        self.k_reaer = 0.4          # 산소 재폭기 계수
        self.O2_demand = 3.0        # 오일 1g 분해 시 산소 3g 소비
        
        # 생태 성장 관련
        self.growth_rate = 0.5      # 플랑크톤 최대 성장률
        self.carrying_capacity = 100.0 
        self.toxicity_threshold = 0.5  # 독성 임계치
        
        # 해안가 민감도 맵 (Biology)
        y, x = np.ogrid[:H, :W]
        self.bio_sensitivity = 0.5 + 0.5 * (x / W)

    def step_1_calculate_hotspots(self, oil_map, toc_map):
        """정화 우선순위 산출: (Oil + 0.5*TOC) * 민감도"""
        norm_oil = oil_map / (np.max(oil_map) + 1e-6)
        norm_toc = toc_map / (np.max(toc_map) + 1e-6)
        chem_score = norm_oil + 0.5 * norm_toc
        priority_map = chem_score * self.bio_sensitivity
        return priority_map

    def step_2_physics_cleaning(self, oil_map, toc_map, priority_map, ships=3, capacity=10):
        """물리적 정화 실행 (Hotspot 타격)"""
        cleaned_oil = oil_map.copy()
        cleaned_toc = toc_map.copy()
        
        flat_priority = priority_map.flatten()
        target_indices = np.argpartition(flat_priority, -ships * capacity)[-ships * capacity:]
        
        removal_eff = 0.9 # 90% 제거 효율
        ys, xs = np.unravel_index(target_indices, (self.H, self.W))
        
        cleaned_oil[ys, xs] *= (1 - removal_eff)
        cleaned_toc[ys, xs] *= (1 - removal_eff)
        
        cleaning_mask = np.zeros_like(oil_map)
        cleaning_mask[ys, xs] = 1.0
        
        return cleaned_oil, cleaned_toc, cleaning_mask

    def step_3_recovery_odes(self, oil_map, do_map, plankton_map, dt_days=1.0):
        """회복 시뮬레이션 (미분방정식 풀이)"""
        # 1. Oil Degradation
        delta_oil = self.k_deg_natural * oil_map * dt_days
        next_oil = np.clip(oil_map - delta_oil, 0, None)
        
        # 2. DO Sag & Recovery
        consumption = delta_oil * self.O2_demand
        dampening = 1.0 / (1.0 + 10.0 * oil_map)
        reaeration = self.k_reaer * dampening * (8.0 - do_map) * dt_days
        next_do = np.clip(do_map - consumption + reaeration, 0, 8.0)
        
        # 3. Plankton Recovery
        toxicity_inhibition = np.clip(oil_map / self.toxicity_threshold, 0, 1.0)
        growth_factor = (1.0 - plankton_map / self.carrying_capacity)
        effective_growth = self.growth_rate * growth_factor * (1.0 - toxicity_inhibition)
        next_plankton = plankton_map + (effective_growth * plankton_map * dt_days)
        
        next_chla = next_plankton * 0.05
        return next_oil, next_do, next_plankton, next_chla

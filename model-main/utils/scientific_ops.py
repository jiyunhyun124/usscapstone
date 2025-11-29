# utils/scientific_ops.py (심화 버전: Reaction-Diffusion)
import numpy as np
from scipy.ndimage import laplace # [NEW] 확산 계산용 라이브러리

class ScientificCleanupRecovery:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        
        # [Scientific Constants]
        self.k_deg_natural = 0.05
        self.k_deg_active = 0.2
        self.k_reaer = 0.4
        self.O2_demand = 3.0
        
        self.growth_rate = 0.5
        self.carrying_capacity = 100.0 
        self.toxicity_threshold = 0.5
        
        # [NEW] 확산 계수 (Diffusion Coefficients)
        # 물질이 옆 칸으로 퍼지는 속도 (물리적 혼합)
        self.D_oil = 0.01      # 오일은 끈적해서 느리게 퍼짐
        self.D_oxygen = 0.1    # 산소는 물속에서 잘 퍼짐
        self.D_plankton = 0.05 # 플랑크톤은 물 흐름 따라 이동

        # 민감도 맵
        y, x = np.ogrid[:H, :W]
        self.bio_sensitivity = 0.5 + 0.5 * (x / W)

    def step_1_calculate_hotspots(self, oil_map, toc_map):
        """정화 우선순위 산출"""
        norm_oil = oil_map / (np.max(oil_map) + 1e-6)
        norm_toc = toc_map / (np.max(toc_map) + 1e-6)
        chem_score = norm_oil + 0.5 * norm_toc
        priority_map = chem_score * self.bio_sensitivity
        return priority_map

    def step_2_physics_cleaning(self, oil_map, toc_map, priority_map, ships=3, capacity=10):
        """물리적 정화 실행"""
        cleaned_oil = oil_map.copy()
        cleaned_toc = toc_map.copy()
        
        flat_priority = priority_map.flatten()
        target_indices = np.argpartition(flat_priority, -ships * capacity)[-ships * capacity:]
        
        removal_eff = 0.9 
        ys, xs = np.unravel_index(target_indices, (self.H, self.W))
        
        cleaned_oil[ys, xs] *= (1 - removal_eff)
        cleaned_toc[ys, xs] *= (1 - removal_eff)
        
        cleaning_mask = np.zeros_like(oil_map)
        cleaning_mask[ys, xs] = 1.0
        
        return cleaned_oil, cleaned_toc, cleaning_mask

    # [UPGRADED] 심화된 회복 시뮬레이션 (반응-확산 모델)
    def step_3_recovery_odes(self, oil_map, do_map, plankton_map, dt_days=1.0):
        """
        PDE Solver: Reaction(반응) + Diffusion(확산)
        기존의 단순 감소뿐만 아니라, 주변의 깨끗한 물이 섞이는 현상까지 시뮬레이션
        """
        # 1. 확산 항 (Diffusion Term: D * ∇²C)
        # 주변 농도 차이에 따라 물질이 이동함
        diff_oil = self.D_oil * laplace(oil_map)
        diff_do = self.D_oxygen * laplace(do_map)
        diff_plank = self.D_plankton * laplace(plankton_map)
        
        # 2. 반응 항 (Reaction Term: 기존 과학적 수식)
        # (A) Oil Decay (1차 반응)
        # delta_oil은 '줄어드는 양'
        delta_oil = self.k_deg_natural * oil_map 
        
        # (B) DO Sag (스트리터-펠프스)
        consumption = delta_oil * self.O2_demand
        dampening = 1.0 / (1.0 + 10.0 * oil_map)
        reaeration = self.k_reaer * dampening * (8.0 - do_map)
        # 산소 변화량 = 재폭기 - 소비
        react_do = reaeration - consumption
        
        # (C) Plankton Growth (로지스틱 + 독성)
        tox_inhib = np.clip(oil_map / self.toxicity_threshold, 0, 1.0)
        growth = self.growth_rate * (1 - plankton_map/self.carrying_capacity) * (1 - tox_inhib)
        # 플랑크톤 변화량 = 성장률 * 현재양
        react_plank = growth * plankton_map

        # 3. 최종 업데이트 (Current + Diffusion*dt + Reaction*dt)
        # 오일은 줄어들고(Reaction -), 퍼짐(Diffusion +)
        next_oil = oil_map + (diff_oil - delta_oil) * dt_days
        
        # 산소와 플랑크톤은 반응 결과와 확산을 더함
        next_do = do_map + (diff_do + react_do) * dt_days
        next_plank = plankton_map + (diff_plank + react_plank) * dt_days
        
        # 물리적 한계 설정 (Clipping)
        next_oil = np.clip(next_oil, 0, None)
        next_do = np.clip(next_do, 0, 8.0)
        next_plank = np.clip(next_plank, 0, None)
        
        # (시각화용) 엽록소-a는 플랑크톤의 5%
        next_chla = next_plank * 0.05
        
        return next_oil, next_do, next_plank, next_chla

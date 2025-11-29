# run_impact_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.scientific_ops import ScientificCleanupRecovery

def main():
    print(">>> [Impact Analysis] AI 도입 효과 비교 분석 시작...")

    # 1. 공통 초기 조건 설정 (2007년 태안 사고 재현)
    H, W = 64, 64
    simulator = ScientificCleanupRecovery(H, W)
    
    # 태안 앞바다 대규모 유출 상황 가정
    y, x = np.ogrid[:H, :W]
    initial_oil = np.exp(-((x - 20)**2 + (y - 32)**2) / (2 * 10.0**2))
    # 초기 산소 고갈 상태
    initial_do = np.full((H, W), 8.0) - (initial_oil * 6.0)
    initial_do = np.clip(initial_do, 0, 8.0)

    # -------------------------------------------------------------------------
    # [Scenario A] 기존 방식 (Traditional / Control Group)
    # - 전략: 핫스팟을 정확히 모르니 넓은 구역을 얕게 청소 (효율 낮음)
    # - 복원: 자연 분해(Natural Attenuation)에만 의존
    # -------------------------------------------------------------------------
    print(">>> 시나리오 A: 기존 방식 시뮬레이션 중...")
    
    # 정화: 핫스팟을 특정하지 못해 전체적으로 20%만 감소시켰다고 가정
    oil_A = initial_oil * 0.8 
    do_A = initial_do.copy()
    plank_A = np.full((H, W), 100.0) # 플랑크톤 로직용 (여기선 그래프엔 DO만 표시)

    history_A = []
    
    # 21일간(3주) 변화 관찰
    for _ in range(21):
        # 복원: 자연 분해율(k_deg_natural) 사용 -> 느림
        # simulator 클래스의 파라미터를 잠시 조작해서 자연 상태로 둠
        temp_k = simulator.k_deg_natural 
        
        # 오일 감소 계산
        delta = temp_k * oil_A
        oil_A = np.clip(oil_A - delta, 0, None)
        
        # 산소 회복 계산
        consumption = delta * simulator.O2_demand
        dampening = 1.0 / (1.0 + 10.0 * oil_A)
        reaeration = simulator.k_reaer * dampening * (8.0 - do_A)
        do_A = np.clip(do_A - consumption + reaeration, 0, 8.0)
        
        history_A.append(np.mean(do_A))

    # -------------------------------------------------------------------------
    # [Scenario B] AI 도입 방식 (AI-Driven / Experimental Group)
    # - 전략: 핫스팟(Hotspot)을 찾아내 집중 타격 (효율 높음)
    # - 복원: 미생물 제제 투입 (Biostimulation) -> 분해 속도 가속
    # -------------------------------------------------------------------------
    print(">>> 시나리오 B: AI 도입 방식 시뮬레이션 중...")
    
    # 1. AI 진단 & 정화 (우선순위 계산 -> 90% 제거)
    # 실제 코드 함수 활용
    toc_map = initial_oil * 0.8 # 가정
    priority = simulator.step_1_calculate_hotspots(initial_oil, toc_map)
    oil_B, _, _ = simulator.step_2_physics_cleaning(initial_oil, toc_map, priority, ships=5)
    
    do_B = initial_do.copy()
    plank_B = np.full((H, W), 100.0)

    history_B = []
    
    # 21일간(3주) 변화 관찰
    for _ in range(21):
        # 복원: 활성 분해율(k_deg_active) 사용 -> 빠름 (4배)
        # 미생물 투입 효과 반영
        temp_k = simulator.k_deg_active 
        
        # 오일 감소
        delta = temp_k * oil_B
        oil_B = np.clip(oil_B - delta, 0, None)
        
        # 산소 회복
        consumption = delta * simulator.O2_demand
        dampening = 1.0 / (1.0 + 10.0 * oil_B)
        reaeration = simulator.k_reaer * dampening * (8.0 - do_B)
        do_B = np.clip(do_B - consumption + reaeration, 0, 8.0)
        
        history_B.append(np.mean(do_B))

    # -------------------------------------------------------------------------
    # [결과 분석] 정량적 효과 계산
    # -------------------------------------------------------------------------
    # 정상 수치(7.5mg/L)에 도달하는 데 걸린 시간 비교
    target_do = 7.5
    
    # Python의 next() 함수로 7.5 넘는 첫 날짜 찾기 (없으면 21일로 간주)
    day_A = next((i for i, v in enumerate(history_A) if v >= target_do), 21)
    day_B = next((i for i, v in enumerate(history_B) if v >= target_do), 21)
    
    improvement = day_A - day_B
    print(f"\n[ANALYSIS REPORT]")
    print(f" - Traditional Recovery Time (to {target_do}mg/L): {day_A} days")
    print(f" - AI-Driven Recovery Time (to {target_do}mg/L):   {day_B} days")
    print(f" => Time Saved: {improvement} days faster!")

    # -------------------------------------------------------------------------
    # [시각화] 비교 그래프 그리기
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    days = range(1, 22)
    plt.plot(days, history_A, 'r--o', label='Traditional (Manual + Natural)', alpha=0.7)
    plt.plot(days, history_B, 'g-^', linewidth=2.5, label='AI-Driven (Hotspot + Bio-enhanced)')
    
    # 골든타임/회복 지점 표시
    plt.axhline(target_do, color='blue', linestyle=':', label='Recovery Threshold (7.5 mg/L)')
    
    # 화살표로 차이 강조
    if day_B < 21:
        plt.annotate(f'{improvement} Days Faster!', 
                     xy=(day_B, target_do), xytext=(day_B+2, target_do-0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title(f"Impact Analysis: Recovery Time Comparison (Taean Scenario)", fontsize=14)
    plt.xlabel("Days after Incident", fontsize=12)
    plt.ylabel("Avg Dissolved Oxygen (mg/L)", fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports", exist_ok=True)
    save_path = "reports/impact_analysis.png"
    plt.savefig(save_path)
    print(f">>> 그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    main()

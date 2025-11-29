# run_scientific_demo.py
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.scientific_ops import ScientificCleanupRecovery  # 방금 만든 파일 불러오기

def main():
    print(">>> 과학적 정화 및 회복 시뮬레이션 시작...")
    
    # 1. 초기 설정
    H, W = 64, 64
    simulator = ScientificCleanupRecovery(H, W)
    
    # 가상 오일 데이터 생성 (AI 예측 결과라고 가정)
    y, x = np.ogrid[:H, :W]
    oil_map = np.exp(-((x - 20)**2 + (y - 32)**2) / (2 * 10.0**2))
    toc_map = oil_map * 0.8
    
    # 초기 생태 상태 (사고 직후)
    do_map = np.full((H, W), 8.0) - (oil_map * 5.0)
    plankton_map = np.full((H, W), 100.0) * (1.0 - oil_map)

    # 2. 단계별 실행
    # Step 1: 우선순위 계산
    priority = simulator.step_1_calculate_hotspots(oil_map, toc_map)
    
    # Step 2: 정화 실행 (배 5척 투입)
    clean_oil, clean_toc, mask = simulator.step_2_physics_cleaning(
        oil_map, toc_map, priority, ships=5
    )
    
    # Step 3: 7일간 회복 시뮬레이션
    history_do = []
    history_plank = []
    curr_oil, curr_do, curr_plank = clean_oil.copy(), do_map.copy(), plankton_map.copy()
    
    for _ in range(7):
        curr_oil, curr_do, curr_plank, _ = simulator.step_3_recovery_odes(
            curr_oil, curr_do, curr_plank
        )
        history_do.append(np.mean(curr_do))
        history_plank.append(np.mean(curr_plank))

    # 3. 결과 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # (지도 그리기 코드 생략 - 이미지는 자동 저장됨)
    axes[0,0].imshow(priority, cmap='Reds'); axes[0,0].set_title("1. Priority Map")
    axes[0,1].imshow(mask, cmap='Greens'); axes[0,1].set_title("2. Cleanup Strategy")
    axes[0,2].imshow(clean_oil, cmap='inferno'); axes[0,2].set_title("3. Post-Cleaning Oil")
    axes[1,0].imshow(curr_do, cmap='Blues'); axes[1,0].set_title("4. Recovered DO")
    axes[1,1].imshow(curr_plank, cmap='Viridis'); axes[1,1].set_title("5. Recovered Plankton")
    
    # 그래프
    ax6 = axes[1,2]
    ax6.plot(range(1,8), history_do, 'b-o', label='DO')
    ax2 = ax6.twinx()
    ax2.plot(range(1,8), history_plank, 'g-x', label='Plankton')
    ax6.set_title("6. Recovery Trend (7 Days)")
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/scientific_result.png")
    print(">>> 결과 저장 완료: reports/scientific_result.png")

if __name__ == "__main__":
    main()

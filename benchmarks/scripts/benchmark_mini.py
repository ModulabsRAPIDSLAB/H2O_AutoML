import os
import socket
import subprocess
import time
import h2o
import h2o.backend
from h2o.automl import H2OAutoML
import openml
import pandas as pd
from pathlib import Path

# 1. WSL 전용 H2O 초기화 함수
def init_h2o_wsl(mem_size="8G"):
    def get_wsl_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    wsl_ip = get_wsl_ip()
    port = "54321"
    jar_path = os.path.join(os.path.dirname(h2o.backend.__file__), "bin", "h2o.jar")

    try:
        h2o.connect(url=f"http://{wsl_ip}:{port}", silent_connect=True)
        print(f"[!] 연결 성공: {wsl_ip}")
    except Exception:
        print(f"[!] 서버 부팅 중 ({wsl_ip})...")
        cmd = ["java", f"-Xmx{mem_size}", "-jar", jar_path, "-ip", wsl_ip, "-port", port]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        h2o.init(url=f"http://{wsl_ip}:{port}")

# 2. 미니 실험 설정 (3개 데이터셋)
mini_datasets = {
    "auc": ["australian", "credit-g"],
    "logloss": ["car"]
}
TIME_STEPS = [1, 3, 5] # 테스트용 (단위: 분)

BASE_DIR = Path(__file__).resolve().parent.parent  # benchmarks 폴더
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True) 

OUTPUT_FILE = "mini_benchmark_results.csv"

def run_mini_benchmark():
    init_h2o_wsl("8G")
    results = []
    
    for m_type, ds_list in mini_datasets.items():
        metric = "AUC" if m_type == "auc" else "logloss"
        
        for ds_name in ds_list:
            print(f"\n>>> [실험 시작] 데이터셋: {ds_name} | 지표: {metric}")
            try:
                # 데이터 로드
                data = openml.datasets.get_dataset(ds_name)
                X, y, _, _ = data.get_data(target=data.default_target_attribute)
                hf = h2o.H2OFrame(pd.concat([X, y], axis=1))
                
                target = data.default_target_attribute
                hf[target] = hf[target].asfactor()
                train, test = hf.split_frame(ratios=[0.8], seed=42)

                row = {"dataset": ds_name, "metric": "roc-auc" if m_type=="auc" else "-logloss"}

                # 시간 단계별 학습
                for t in TIME_STEPS:
                    print(f"    - 예산 {t}분 학습 중...", end="\r")
                    aml = H2OAutoML(max_runtime_secs=t*60, seed=42, sort_metric=metric)
                    aml.train(y=target, training_frame=train)
                    
                    if aml.leader:
                        perf = aml.leader.model_performance(test)
                        score = perf.auc() if m_type=="auc" else -perf.logloss()
                        row[f"{t}min"] = round(score, 4)
                    else:
                        row[f"{t}min"] = "N/A"
                
                results.append(row)
                # 중간 저장
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                
            except Exception as e:
                print(f"\n [!] {ds_name} 오류 발생: {e}")

    # 3. 최종 표 출력
    print("\n\n" + "="*50)
    print("성능 평가 결과 표 (Mini Benchmark)")
    print("="*50)
    final_df = pd.read_csv(OUTPUT_FILE)
    print(final_df.to_markdown(index=False))
    print("="*50)
    print(f"결과가 {OUTPUT_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    run_mini_benchmark()
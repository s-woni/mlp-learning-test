from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import shutil

app = Flask(__name__)
MODEL_PATH = "mlp_single_custom.pkl"
model_pipe = joblib.load(MODEL_PATH)

# 🔧 평균 유통시간 하드코딩
avg_delay_dict = {
    ('ICE', 2): 22.3688,   ('ICE', 3): 17.3992,   ('ICE', 4): 15.981,    ('ICE', 5): 16.0512,
    ('HEV', 2): 22.4633,   ('HEV', 3): 17.2794,   ('HEV', 4): 15.7819,   ('HEV', 5): 14.8167,
    ('EV',  2): 22.5549,   ('EV',  3): 16.5231,   ('EV',  4): 16.1131,   ('EV',  5): 16.9977,
}

# 🔨 입력 피처 구성
def make_feature_row(stage, model_type, option_list):
    if isinstance(option_list, str):
        option_list = json.loads(option_list)

    row = {}
    for m in ["ICE", "HEV", "EV"]:
        row[f"model_type_{m}"] = int(model_type == m)
    for s in range(1, 6):
        row[f"stage_{s}"] = int(stage == s)
    for i in range(1, 6):
        row[f"opt_{i}_only4"] = int(stage == 4 and i in option_list)

    return row

@app.route("/predict_all", methods=["POST"])
def predict_all():
    """
    요청 JSON 예:
    {
      "model_type": "ICE",
      "option_list": [1, 2, 3]
    }
    """
    data = request.get_json()
    model_type = data["model_type"]
    option_list = data.get("option_list", [])

    # 1. 전체 공정 피처 생성
    rows = [make_feature_row(stage, model_type, option_list) for stage in range(1, 6)]
    df = pd.DataFrame(rows)

    # 2. 평균 유통시간 삽입
    df["transport_delay_h"] = [
        avg_delay_dict.get((model_type, stage), 0.0) for stage in range(1, 6)
    ]

    # 3. 누락된 컬럼 보완
    required_cols = (
        [f"model_type_{m}" for m in ["ICE", "HEV", "EV"]] +
        [f"stage_{s}" for s in range(1, 6)] +
        [f"opt_{i}_only4" for i in range(1, 6)] +
        ["transport_delay_h"]
    )
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[required_cols]

    # 4. 예측 수행
    durations = model_pipe.predict(df)

    # 5. 유통시간 더한 결과 반환
    results = []
    total = 0.0
    for stage, base_duration in zip(range(1, 6), durations):
        delay = avg_delay_dict.get((model_type, stage), 0.0)
        total_duration = base_duration + delay
        results.append({
            "stage": stage,
            "pred_duration_h": round(float(base_duration), 3),
            "transport_delay_h": round(delay, 3),
            "total_with_delay_h": round(total_duration, 3)
        })
        total += total_duration

    return jsonify({
        "stages": results,
        "total_duration_h": round(total, 3)
    })

# 🔁 모델 리로드 함수
def reload_model():
    global model_pipe
    model_pipe = joblib.load(MODEL_PATH)
    modified = os.path.getmtime(MODEL_PATH)
    print(f"[Flask] 모델 리로드 완료 - 모델 수정 시각: {datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')}")

# ⬆️ 모델 업로드
@app.route("/upload", methods=["POST"])
def upload_model():
    file = request.files.get("model")
    if not file:
        return jsonify({"error": "파일이 없습니다."}), 400

    # 1. 저장
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("models", exist_ok=True)
    versioned_path = f"models/mlp_model_{now_str}.pkl"
    file.save(versioned_path)

    # 2. 최신 모델로 덮어쓰기
    shutil.copy(versioned_path, MODEL_PATH)
    reload_model()

    # 3. 오래된 파일 삭제 (한 달 기준)
    delete_threshold = now - timedelta(days=30)
    deleted_files = []

    for filename in os.listdir("models"):
        filepath = os.path.join("models", filename)
        if filepath.endswith(".pkl"):
            modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if modified_time < delete_threshold:
                os.remove(filepath)
                deleted_files.append(filename)

    return jsonify({
        "message": "모델 업로드 및 리로드 완료",
        "saved_as": versioned_path,
        "deleted_old_files": deleted_files
    }), 200

# ▶️ 앱 실행
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        reload_model()
    app.run(host="0.0.0.0", port=10000)

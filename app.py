from flask import Flask, request, jsonify
import joblib, json, pandas as pd

app = Flask(__name__)
model_pipe = joblib.load("mlp_single_custom.pkl")

# 모델+공정별 평균 유통시간
avg_delay_dict = {
    ('ICE', 2): 22.3688,   ('ICE', 3): 17.3992,   ('ICE', 4): 15.981,    ('ICE', 5): 16.0512,
    ('HEV', 2): 22.4633,   ('HEV', 3): 17.2794,   ('HEV', 4): 15.7819,   ('HEV', 5): 14.8167,
    ('EV',  2): 22.5549,   ('EV',  3): 16.5231,   ('EV',  4): 16.1131,   ('EV',  5): 16.9977,
}

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
    요청 JSON:
    {
      "model_type": "ICE",
      "option_list": "[1,2,3]"
    }
    """
    data = request.get_json()
    model_type = data["model_type"]
    option_list = data.get("option_list", "[]")
    if isinstance(option_list, str):
        option_list = json.loads(option_list)

    # 전체 공정 1~5에 대한 예측
    rows = [make_feature_row(stage, model_type, option_list) for stage in range(1, 6)]
    df = pd.DataFrame(rows)

    # 누락된 컬럼 채우기
    required_cols = (
        [f"model_type_{m}" for m in ["ICE", "HEV", "EV"]] +
        [f"stage_{s}" for s in range(1, 6)] +
        [f"opt_{i}_only4" for i in range(1, 6)]
    )
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[required_cols]

    # 예측
    durations = model_pipe.predict(df)

    # delay 합산
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

from flask import Flask, request, jsonify
import joblib, json, pandas as pd

app = Flask(__name__)
model_pipe = joblib.load("mlp_single_custom.pkl")

def make_feature_row(payload):
    # payload: {"stage":int, "model_type":str, "option_list":str|list}
    stage      = int(payload["stage"])
    model_type = payload["model_type"]
    opts       = payload.get("option_list", "[]")
    if isinstance(opts, str):
        opts = json.loads(opts)

    # model_type 원-핫, stage 원-핫
    row = {}
    for m in ["ICE","HEV","EV"]:
        row[f"model_type_{m}"] = 1 if model_type == m else 0
    for s in range(1,6):
        row[f"stage_{s}"] = 1 if stage == s else 0

    # 옵션은 오직 4단계에만
    for i in range(1,6):
        row[f"opt_{i}_only4"] = 1 if (stage == 4 and i in opts) else 0

    return pd.Series(row)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    X       = make_feature_row(payload).to_frame().T
    pred    = float(model_pipe.predict(X)[0])
    return jsonify({
        "stage":           payload["stage"],
        "model_type":      payload["model_type"],
        "pred_duration_h": round(pred, 3)
    })

@app.route("/predict_all", methods=["POST"])
def predict_all():
    data       = request.get_json()
    model_type = data["model_type"]
    opts       = data.get("option_list", "[]")

    # 각 stage별 feature row 만들기
    rows = []
    for s in range(1,6):
        payload = {
            "stage":        s,
            "model_type":   model_type,
            "option_list":  opts
        }
        # make_feature_row → pandas.Series
        rows.append(make_feature_row(payload))
    df_feat = pd.DataFrame(rows)

    # 누락 컬럼 보정 (train 때 쓰던 컬럼)
    needed = (
        [f"model_type_{m}" for m in ["ICE","HEV","EV"]] +
        [f"stage_{s}"      for s in range(1,6)] +
        [f"opt_{i}_only4"  for i in range(1,6)]
    )
    for col in needed:
        if col not in df_feat.columns:
            df_feat[col] = 0
    df_feat = df_feat[needed]

    # 예측
    preds = model_pipe.predict(df_feat)

    # 결과 포맷
    results = [
        {"stage": s, "pred_duration_h": round(float(pred), 3)}
        for s, pred in zip(range(1,6), preds)
    ]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 모델 불러오기
model = joblib.load("mlp_multioutput_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    option_1 = data.get("option_1", 0)
    option_2 = data.get("option_2", 0)
    X = pd.DataFrame([[option_1, option_2]], columns=["option_1", "option_2"])
    pred = model.predict(X)[0]
    result = {
        "process_1_days": round(pred[0], 1),
        "process_2_days": round(pred[1], 1),
        "process_3_days": round(pred[2], 1),
        "process_4_days": round(pred[3], 1),
        "process_5_days": round(pred[4], 1),
        "total_days": round(sum(pred), 1)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

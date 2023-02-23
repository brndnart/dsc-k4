from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from

import pickle, re
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info={
        "title": LazyString(lambda: "API Documentation for Deep Learning"),
        "version": LazyString(lambda: "1.0.0"),
        "description": LazyString(lambda: "Dokumentasi API untuk Deep Learning"),
    },
    host=LazyString(lambda: request.host),
)

swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "docs", "route": "/docs.json"}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs",
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

sentiment = ["negative", "neutral", "positive"]

def cleansing(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    return text

# Neural Network Feature and Model

neural_network_feature_file = open("./neural_network/feature.p", "rb")
neural_network_model_file = open("./neural_network/model.p", "rb")

neural_network_feature = pickle.load(neural_network_feature_file)
neural_network_model = pickle.load(neural_network_model_file)

neural_network_feature_file.close()
neural_network_model_file.close()

# Neural Network API

@swag_from("./docs/neural_network.yml", methods=["POST"])
@app.route("/neural_network", methods=["POST"])
def neural_network():
    text_input = request.form.get("text")
    text = [cleansing(text_input)]

    feature = neural_network_feature.transform(text)
    get_sentiment = neural_network_model.predict(feature)[0]

    json_res = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis using Neural Network",
        "data": {"text": text[0], "sentiment": get_sentiment},
    }

    res_data = jsonify(json_res)

    return res_data

# LSTM Feature and Model

lstm_feature_file = open("./lstm/x_pad_sequences.pickle", "rb")
lstm_feature = pickle.load(lstm_feature_file)
lstm_feature_file.close()

lstm_tokenizer_file = open("./lstm/tokenizer.pickle", "rb")
lstm_tokenizer = pickle.load(lstm_tokenizer_file)
lstm_tokenizer_file.close()

lstm_model = tf.keras.models.load_model("./lstm/model.h5")

# LSTM API

@swag_from("./docs/lstm.yml", methods=["POST"])
@app.route("/lstm", methods=["POST"])
def lstm():
    text_input = request.form.get("text")
    text = [cleansing(text_input)]

    feature = lstm_tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=lstm_feature.shape[1])

    prediction = lstm_model.predict(feature)
    print(prediction[0])
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_res = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis using LSTM",
        "data": {"text": text[0], "sentiment": get_sentiment},
    }

    res_data = jsonify(json_res)

    return res_data

if __name__ == "__main__":
    app.run()

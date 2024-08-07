from split_model import CloudInfer
from flask import Flask, request

app = Flask(__name__)

cloud_infer = CloudInfer(
    'runs/train/vae_one_branch_te_r4/weights/cloud.pt', device='2')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        pred = cloud_infer.run(data)
        print(pred)
        return '1'


if __name__ == '__main__':
    app.run()

# FLASK_ENV=development FLASK_APP=app.py flask run

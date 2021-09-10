import datetime

import cv2
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from sqlalchemy import func
from db_connect import db
from models import Target, Flower

app =Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:duddnr1229@localhost:3306/ai_college"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

db.init_app(app)


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')



@app.route("/image", methods=["GET", "POST"])
def input_image():
    if request.method == "POST":
        print(request.files)
        try:
            if 'input-image' in request.files:
                imageFile = request.files['input-image'].read()

                # 전처리
                IMG_SIZE = 150
                npimg = np.fromstring(imageFile, np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = np.array(img)
                X = []
                X.append(img)
                X = np.array(X)
                X = X / 255

                # 입력 받은 이미지 예측
                expred = model.predict(X)
                expred_digits = np.argmax(expred, axis=1)

                # 분류
                answer = expred_digits
                if answer == 0:
                    ans = "백합_시베리아"
                if answer == 1:
                    ans = "안개_오버타임"
                if answer == 2:
                    ans = "용담_용담"
                if answer == 3:
                    ans = "카네이션_레드(스프레이)"
                if answer == 4:
                    ans = "해바라기_해바라기"

                # 문자열 스플릿
                fl_item, fl_type = str(ans).split('_')
                print(fl_item, fl_type)
                flist = Target(fl_item, fl_type)
                db.session.add(flist)
                db.session.commit()
                print(flist.fl_type, flist.fl_item)

                flower_dict = []

                today = datetime.datetime.today().strftime("%Y-%m-%d")
                fcost = db.session.query(Flower.poomname, Flower.goodname, Flower.lvname, func.sum(Flower.qty).label('qty'), func.avg(Flower.cost).label('cost'), Flower.dateinfo ).\
                    filter(Flower.poomname == flist.fl_item, Flower.goodname == flist.fl_type).\
                    filter(today >= func.ADDDATE(Flower.dateinfo, -7)).\
                    group_by(Flower.lvname).all()

                assert isinstance(fcost, object)
                for f in fcost:
                    flower_dict.append({
                        "poomname": f.poomname,
                        "goodname": f.goodname,
                        "lvname": f.lvname,
                        "cost": int(f.cost),
                        "qty": int(f.qty),
                        "dateinfo": str(f.dateinfo)
                    })
                print(flower_dict)

        except Exception as e:
            print(e)

        return jsonify(flower_dict)


if __name__ == '__main__':
    # ml/project_code_final.py 선 실행 후 생성
    model = load_model('./ml/model.h5')

    # Flask 서비스 스타트
    app.run(debug=True)

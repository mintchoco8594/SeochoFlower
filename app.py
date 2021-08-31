import os

import cv2
import pymysql
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from db_connect import db
from models import Target, Flower


#db = pymysql.connect(host='localhost', port=3306, user='root', passwd='duddnr1229', db='ai_college', charset="utf8")

#app.config['JSON_AS_ASCII'] = False

app =Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:duddnr1229@localhost:3306/ai_college"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

db.init_app(app)

@app.route('/')
def index():
    return render_template("draft_main.html")

@app.route("/", methods=["POST"])
def image():
    if request.method == 'POST':

        # 업로드 파일 처리
        file = request.files['input-image']
        f_name = file.filename
        file.save('static/saved_file/' + secure_filename(f_name))

        if not file:
            return render_template('draft_main.html', label="No Files")

        IMG_SIZE = 150
        # convert string data to numpy array
        img = cv2.imread('static/saved_file/' + secure_filename(f_name), cv2.IMREAD_COLOR)
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
            ans = "백합_글로벌하모니"
        if answer == 4:
            ans = "해바라기_해바라기"

        # 품목, 품종 스플릿
        fl_item, fl_type = str(ans).split('_')
        print(fl_item, fl_type)
        flist = Target(fl_item,fl_type)
        db.session.add(flist)
        db.session.commit()
        print(flist.fl_type, flist.fl_item)

        flower_dict = []
        fcost = db.session.query(Flower).all()
        for f in fcost:
            flower_dict.append({
                "poomname": f.poomname,
                "goodname" : f.goodname,
                "lvname" : f.lvname,
                "cost": f.cost,
                "qty" : f.qty
            })
        print(flower_dict)
        return jsonify(flower_dict)
        # with db.cursor() as cur:
        #     cur.execute(
        #         "select poomname, goodname, lvname, round(avg(cost)), sum(qty) from realtime_flower where poomname = '%s' and goodname = '%s' group by lvname" \
        #         % (fl_item, fl_type))
        #     rows = cur.fetchall()  # 데이터저장
        #     print(rows)
        #     flower_dict = []
        #     for row in rows:
        #         flower_dict.append({
        #          "poomname" : str(row[0]),
        #          "goodname" : str(row[1]),
        #          "lvname" : str(row[2]),
        #          "cost" : str(row[3]),
        #          "qty" : str(row[4])
        #         })
        #     #### json 형식으로 리턴해주기 #####
        #     return jsonify(flower_dict)
if __name__ == '__main__':
    # ml/project_code_final.py 선 실행 후 생성
    model = load_model('C:/Users/ukiii/PycharmProjects/pythonProject3/ml/model.h5')

    # Flask 서비스 스타트
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8000, debug=True)
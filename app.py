"""
Run a rest API exposing the yolov5s object detection model
"""
import datetime
import detect
import io
import os
import pandas as pd
import pymysql

from sqlalchemy import func
from db_connect import db
from db_models import Target, Flower

import shutil
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# db = pymysql.connect(host='localhost', port=3306, user='root', passwd='1234', db='ai_college', charset="utf8")
# cur = db.cursor()  # 커서 클래스 호출

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:1234@localhost:3306/ai_college"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

db.init_app(app)

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/image_detect", methods=["GET", "POST"])
def detect_image():
    if request.method == "POST":
        print(request.files)
        try:
            if 'input-image' in request.files:
                imageFile = request.files['input-image']

            fileName = imageFile.filename
            # 이미지파일 임시 저장
            savePath = "datasets/images/train/"  # 매번 비워지는 임시 저장소
            savePath2 = "datasets/images/val/"   # 누적 저장

            if os.path.exists(savePath):
                shutil.rmtree(savePath)  # 파일 삭제
            if not os.path.exists(savePath):
                os.mkdir(savePath)       # 디렉토리 생성

            # 이미지 디텍션을 위한 임시 저장
            imageFile.save(savePath + secure_filename(fileName))

            # 이미지 디텍션 실행
            detect.run()

            with open(savePath + secure_filename(fileName), 'rb') as f:
                fileRead = f.read()
            img = Image.open(io.BytesIO(fileRead))
            results = model(img, size=640)  # reduce size=320 for faster inference

            img.save(savePath2 + secure_filename(fileName))

        except Exception as e:
            print(e)

        result_list = []
        json_data = pd.DataFrame(results.pandas().xyxy[0], columns=["class"])

        for i in range(0, len(json_data)):
            result_list.append(json_data.values[i][0])

        # 중복 제거
        result_list = set(result_list)
        result_list = list(result_list)

        print(result_list)

        dict={0:'거베라_거베라',
            1: '국화_코스모스',
            2: '국화_디스버드',
            3: '국화_백선',
            4: '튤립_레드',
            5: '튤립_로얄버진',
            6: '백합_메두사',
            7: '백합_시베리아',
            8: '수국_그린',
            9: '수국_핑크',
            10: '데이지_데이지',
            11: '데이지_겹',
            12: '용담_용담',
            13: '장미_푸에고',
            14: '장미_하젤',
            15: '카네이션_핑크',
            16: '카네이션_레드(스프레이)',
            17: '안개_오버타임',
            18: '해바라기_겹',
            19: '해바라기_해바라기',
            20: 'out_of_distribution'}

        flower_list = []
        for i in range(0,len(result_list)):
            flower_list.append(dict[result_list[i]])
        print(flower_list)
        return jsonify(flower_list)


@app.route("/analyse_result", methods=["GET"])
def analyse_image():
    # 꽃 품목_품종 가져오기
    fl_name = request.args.get("flower_name")
    print(fl_name)

    #문자열 스플릿
    fl_item, fl_type = str(fl_name).split('_')

    flist = Target(fl_item, fl_type)
    db.session.add(flist)
    db.session.commit()
    print(flist.fl_item, flist.fl_type)

    # 딕셔너리에 담아서 json 형태로 변환하여 전송
    flower_dict = []

    today = datetime.datetime.today().strftime("%Y-%m-%d")
    fcost = db.session.query(Flower.poomname, Flower.goodname, Flower.lvname, func.sum(Flower.qty).label('qty'),
                             func.avg(Flower.cost).label('cost'), Flower.dateinfo). \
        filter(Flower.poomname == flist.fl_item, Flower.goodname == flist.fl_type). \
        filter(today >= func.ADDDATE(Flower.dateinfo, -7)). \
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

    # cur.execute(
    #     # "SELECT * FROM realtime_flower where poomname = '%s' and goodname = '%s' order by qty" % (
    #     # fl_item, fl_type))  # 모델 결과값에 따라 쿼리문 작성하는 부분
    #     "SELECT poomname, goodname, lvname, qty, round(max(cost))"
    #     " FROM realtime_flower where poomname = '%s' and goodname = '%s' group by lvname;" % (
    #         fl_item, fl_type))  # 모델 결과값에 따라 쿼리문 작성하는 부분
    # rows = cur.fetchall()  # 데이터저장
    # print(rows)

    # 딕셔너리에 담아서 json 형태로 변환하여 전송
    # flower_dict = []
    # for row in rows:
    #     flower_dict.append({
    #         "poomname": row[0],
    #         "goodname": row[1],
    #         "lvname": row[2],
    #         "qty": row[3],
    #         "cost": row[4]
    #     })

    return jsonify(flower_dict)


if __name__ == "__main__":
    # hubconf.py 경로, best.pt 경로
    model = torch.hub.load('C:/Users/FineTiger/PycharmProjects/flowerProject1', 'custom', path='best.pt', source='local')  # local repo
    app.run(debug=True)  # debug=True causes Restarting with stat

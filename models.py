from db_connect import db

class Target(db.Model):
    __tablename__ = 'target'
    id = db.Column(db.Integer, primary_key=True,
                       nullable=False, autoincrement=True)
    fl_item = db.Column(db.String(256),nullable=False)
    fl_type = db.Column(db.String(256), nullable=False)

    def __init__(self, fl_item, fl_type):
        self.fl_item = fl_item
        self.fl_type = fl_type


class Flower(db.Model):
    __tablename__ = 'realtime_flower'
    id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True)
    poomname = db.Column(db.String(100), nullable=False)
    goodname = db.Column(db.String(100), nullable=False)
    lvname = db.Column(db.String(100), nullable=False)
    qty = db.Column(db.Integer(), nullable=False)
    cost = db.Column(db.Integer(), nullable=False)

    def __init__(self, poomname, goodname, lvname, qty, cost):
        self.poomname = poomname
        self.email = goodname
        self.phone = lvname
        self.qty = qty
        self.cost = cost

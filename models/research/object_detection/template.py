# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import subprocess
import json
from time import sleep
import object_detection_tutorial_flask as odt
app = Flask(__name__) 

#결과를 보여주기 위한 함수
@app.route('/result') 
def result():
    dict = {'dam':50, 'mul':60, 'math':70}
    return render_template('template.html', result=dict)

#자동으로 계산을 처리하기 위한 함수
@app.route('/run')
def run():
    #odt 모듈에 들어있는 calculate함수를 사용하므로써 계산 속도를 높인다.
    odt.calculate()
    product_result = None
    with open("product_file.json", "r") as pt_json:
        product_result = json.load(pt_json)
        return render_template('run.html', result = product_result)
        
if __name__ == '__main__':
    #db에서 상품을 끌어올림
    odt.start_db()
    odt.load_from_db()
    #필요한 상품정보를 불러온다.
    odt.init_program()
    app.run(debug=True)
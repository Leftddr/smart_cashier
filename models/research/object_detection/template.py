from flask import Flask, request, render_template
import subprocess
import json
from time import sleep
app = Flask(__name__) 

#결과를 보여주기 위한 함수
@app.route('/result') 
def result():
    dict = {'dam':50, 'mul':60, 'math':70}
    return render_template('template.html', result=dict)

#자동으로 계산을 처리하기 위한 함수
@app.route('/run')
def run():
    command = "python object_detection_tutorial.py"
    p = subprocess.Popen(command,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE,
                         shell = True)
    while p.poll() is None:
        print('Loading...')
        sleep(.1)
    #에러처리
    if p.returncode != 0:
        print("Error : " + str(p.stderr.read()))
    
    product_result = None
    with open("product_file.json", "r") as pt_json:
        product_result = json.load(pt_json)
        return render_template('run.html', result = product_result)
        
if __name__ == '__main__':
    app.run(debug=True)
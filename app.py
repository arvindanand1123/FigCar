
# coding: utf-8

# In[4]:


from flask import Flask
from flask import request
import os


# In[5]:
app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to Index'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST' and request.files['image']:
        f = request.files['image']
        f.save('/home/arvind_anand1123/Workspace/FigCar/Input/ImageFile/pic.jpg')
        import CNN_Model_Test as fc
        result = fc.CNN()
        print(result)
        return result
    else:
        return "Didn't post"

@app.route('/out', methods=['GET'])
def download():
    if request.method == 'GET':
        import CNN_Model_Test as fc
        return fc.CNN()
    else:
        return "Didn't hit"

# In[6]:

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 2222)
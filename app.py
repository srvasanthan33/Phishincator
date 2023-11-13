from flask import Flask, render_template, request, url_for
import pickle

app = Flask(__name__)

# Load the models
with open('pipeline_with_LR_Tfid.pkl', 'rb') as model_file:
    loaded_LR = pickle.load(model_file)
with open('pipeline_with_MNB_Tfid.pkl', 'rb') as model_file:
    loaded_MNB = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method == 'GET':
        return render_template('model_selection.html')
    
    # post method 
    selected_model = request.form.get('model')
    url = request.form.get('url')
    if selected_model == 'lr':
        prediction_result = loaded_LR.predict([url])[0]
    elif selected_model == 'mnb':
        prediction_result = loaded_MNB.predict([url])[0]
    
    sel_model = 'Logistic Regression' if selected_model == 'lr' else 'MultinomialNB'
    return render_template('result.html',url=url,model=sel_model,result=prediction_result)
    

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
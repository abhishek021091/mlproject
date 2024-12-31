from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

application = Flask(__name__, static_folder='static', template_folder='templates')

app=application

## Route for a home page


@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')


        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()

        result = predict_pipeline.predict(pred_df)
        
        if result[0] < 100 and result[0] >= 0:
            pass
        elif result[0] > 100:
            result[0] = 100
        elif result[0]<0:
            result[0] = 0
        
        return render_template('home.html',results=result[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
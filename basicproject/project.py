from flask import Flask ,render_template,request,jsonify
import pandas as pd
import pickle as pk
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("birth.html")

def get_cleaned_data(form_data):
    #the data we enter at frontend in come in form data 
    #we cleaned means acccess the data in variable 

    gestation=float(form_data["gestation"])
    parity=int(form_data["parity"])
    age=float(form_data["age"])
    height=float(form_data["height"])
    weight=float(form_data["weight"])
    smoke=int(form_data["smoke"])

    #store the data in cleaned data dict for returning in json format.
    #we train our model on 2d data so we pass in the list format otherwise show error
    # #ValueError: If using all scalar values, you must pass an index
    cleaned_data={
        "gestation":[gestation],
        "parity":[parity],
        "age":[age],
        "height":[height],
        "weight":[weight],
        "smoke":[smoke]
    }

    return cleaned_data

@app.route("/predict",methods=["POSt"])

def predictbirth():
    #collect the info from ui
    baby_data=request.form
    #calling function
    baby_data_form=get_cleaned_data(baby_data)
    #making data frame
    baby_df=pd.DataFrame(baby_data_form)
    #opening the save model
    with open("model/model.pkl","rb") as obj:
        mymodel=pk.load(obj)

    prediction=mymodel.predict(baby_df)
    prediction=round(float(prediction[0]),2)
    
    response={

        "prediction":prediction
    }

    return render_template("birth.html",prediction=prediction)




if __name__=="__main__":
    app.run(debug=True)
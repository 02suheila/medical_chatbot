from flask import Flask, render_template, request
from flask import Flask, render_template, request, jsonify
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv

app = Flask(__name__)

# Your existing code for the chatbot
# ...
# Load necessary data and models
training = pd.read_csv(r"C:\Users\ROOHI\Documents\chatbot_HC\data\Training.csv")
cols = training.columns
cols = cols[:-1]
le = preprocessing.LabelEncoder()
le.fit(training['prognosis'])
clf = DecisionTreeClassifier()
clf.fit(training[cols], le.transform(training['prognosis']))
model = SVC()
model.fit(training[cols], le.transform(training['prognosis']))

severityDictionary = {}
description_list = {}
precautionDictionary = {}



def getSeverityDict():
    global severityDictionary
    with open(r"C:\Users\ROOHI\Documents\chatbot_HC\master_data\Symptom_severity.csv") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getDescription():
    global description_list
    with open(r"C:\Users\ROOHI\Documents\chatbot_HC\master_data\symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getprecautionDict():
    global precautionDictionary
    with open(r"C:\Users\ROOHI\Documents\chatbot_HC\master_data\symptom_precaution.csv") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)



getSeverityDict()
getDescription()
getprecautionDict()


def sec_predict(symptoms_exp):
    df = pd.read_csv(r"C:\Users\ROOHI\Documents\chatbot_HC\data\Training.csv")
    X = df.iloc[:, :-1]
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X, df['prognosis'])
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    return le.inverse_transform(rf_clf.predict([input_vector]))


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))



def tree_to_code(tree, feature_names, disease_input, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    def recurse(node, depth):
        nonlocal symptoms_present
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = training.columns
            result_data = []

            for disease in present_disease:
                # Check if the disease is in the training set
                if disease in red_cols:
                    symptoms_given = red_cols[training.loc[disease].values[0].nonzero()]
                    print("Are you experiencing any ")
                    symptoms_exp = []
                    for syms in list(symptoms_given):
                        inp = ""
                        print(syms, "? : ", end='')
                        while True:
                            inp = input("")
                            if inp == "yes" or inp == "no":
                                break
                            else:
                                print("provide proper answers i.e. (yes/no) : ", end="")
                        if inp == "yes":
                            symptoms_exp.append(syms)

                    second_prediction = sec_predict(symptoms_exp)
                    result = calc_condition(symptoms_exp, num_days)

                    result_data.append({
                        "disease": disease,
                        "description": description_list[disease],
                        "precautions": precautionDictionary[disease],
                        "result": result,
                    })
                else:
                    # Handle the case when the disease is not in the training set
                    result_data.append({
                        "disease": disease,
                        "description": "No information available",
                        "precautions": [],
                        "result": "Please consult a doctor for more information.",
                    })
            print(result_data)
            return result_data

    return recurse(0, 1)
def chatbot_function(symptoms, num_days, para):
    # Placeholder logic, replace with your actual chatbot code
    result = f"Chatbot Result: Symptoms - {symptoms}, Number of Days - {num_days},pre-{num_days}"
    return result



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the form submission and call your chatbot function
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')  # Adjust the form field names as needed
        num_days = int(request.form.get('num_days')) 
        para = int(request.form.get('num_days')) # Adjust the form field names as needed

        # Call your chatbot function with the symptoms and num_days
        result = chatbot_function(symptoms, num_days ,para)
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)














    

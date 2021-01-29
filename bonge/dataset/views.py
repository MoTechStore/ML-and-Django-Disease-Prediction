from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from django.http import JsonResponse
from django.http import HttpResponse
from pandas import read_csv
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib as joblib
import argparse
import os



disease=['Anemia','Preeclampsia','Gestational Diabetes','placenta pevia','Fetal Problems','Hyperemesis gravidurum','AIDS','Diabetes ',
'Yeast Infection','Listeriosis','Hypertension ','Respiratory Infection','Cervical spondylosis','Ectopic pregnancy',
'Preterm labor','Miscarriage','Urinary tract infection'
]


def home(request):
    symptoms =['vaginal_rash','protein_in_urine','swollen_vagina','vaginal_discharge','pelvic_pain','vaginal_itching','continuous_sneezing','shivering','chills','joint_pain',
'stomach_pain','muscle_pain','vomiting','burning_vagina','abnormal_foetus_movement','foetus_smaller_than_normal',
'fatigue','weight_gain','mood_swings','weight_loss','patches_in_throat','irregular_sugar_level','cough','high_fever','breathlessness','sweating',
'dehydration','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','swollen_hands_face_and_feets',
'back_pain','abdominal_pain','diarrhoea','yellow_urine','vision_problem','excessive_vagina_bleeding','acute_liver_failure','swelling_of_stomach','history_of_alcohol_consumption',
'swelled_lymph_nodes','belly_pain','continuous_feel_of_urine','loss_of_smell','look_pale','acute_liver_failure','skin_peeling','chest_pain',
'fast_heart_rate','bloody_stool','dizziness','obesity','puffy_face_and_eyes','excessive_hunger','extra_marital_contacts','loss_of_smell','bladder_discomfort','continuous_feel_of_urine','belly_pain','increased_appetite','family_history','history_of_alcohol_consumption','skin_peeling']

    symptoms = sorted(symptoms)
    context = {'symptoms': symptoms}
    
    if request.method == 'POST':
      Symptom1 = request.POST["Symptom1"]
      Symptom2 = request.POST["Symptom2"]
      Symptom3 = request.POST["Symptom3"]
      Symptom4 = request.POST["Symptom4"]
      Symptom5 = request.POST["Symptom5"]

      psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]
      print("Received Input From User")
      print(psymptoms)

# Training dataset
      data = pd.read_csv(os.path.join(r'C:\Users\MASHALO\Desktop\ujasi\bonge\dataset', 'risk_disease_prediction.csv'))
      print(len(data))
      data.replace({'prognosis': {'Anemia':0,'Preeclampsia':1,'Gestational Diabetes':2,'placenta pevia':3,'Fetal Problems':4,'Hyperemesis gravidurum':5,'AIDS':6,
'Yeast Infection':7,'Listeriosis':8,'Hypertension':9,'Respiratory Infection':10,'Cervical spondylosis':11,'Ectopic pregnancy':12,
'Preterm labor':13,'Miscarriage':14,'Urinary tract infection':15}}, inplace=True)
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# for i in data.columns:
#     if i!='Age' and i!='prognosis':
#         dt[i]=le.fit_transform(dt[i])


# df_x = data.iloc[:,0:62]
# df_y = data.iloc[:,-1]


  # x_train, x_test, y_train,y_test = train_test_split(df_x,df_y, test_size=0.4, random_state=42)
  # from sklearn.naive_bayes import MultinomialNB
  # clf = MultinomialNB()
  # clf.fit(x_train, y_train)
  # clf.score(x_test, y_test)
  # from sklearn.model_selection import cross_val_score
  # v = cross_val_score(clf, X_train, y_train, cv=5)

  # for i in range(5):
  #     print("Accuracy Of NB is : {0:2%}".format(v[i,]))
  #     print("Mean Accuracy of NB is : ", v.mean())

   l2 = []
   for x in range(0, len(symptoms)):
     l2.append(0)
     print("Am printing L1")
     print(l2)

   df_x = data['prognosis']
   print(df_x.head())
  
  print("Feature Selection On Training Dataset")
  X = data[symptoms]
  y = data[["prognosis"]]
  np.ravel(y)
  print("X Features")
  print(X.head())
  print("y Features")
  print(y.head())


# Testing dataset
      dataset = pd.read_csv(os.path.join(r'C:\Users\MASHALO\Desktop\ujasi\bonge\dataset', 'risk_disease_prediction.csv'))
      print("Length Of Training Dataset")
      print(len(dataset))
      
      dataset.replace({'prognosis': {'Anemia':0,'Preeclampsia':1,'Gestational Diabetes':2,'placenta pevia':3,'Fetal Problems':4,'Hyperemesis gravidurum':5,'AIDS':6,
'Yeast Infection':7,'Listeriosis':8,'Hypertension':9,'Respiratory Infection':10,'Cervical spondylosis':11,'Ectopic pregnancy':12,
'Preterm labor':13,'Miscarriage':14,'Urinary tract infection':15}}, inplace=True)

      print("Feature Selection On Testing Dataset")
      X_test = dataset[symptoms]
      print("Testing X Features")
      print(X_test.head())
      y_test = dataset[["prognosis"]]
      print("Testing Y Features")
      print(y_test.head())
      np.ravel(y_test)

      clf = tree.DecisionTreeClassifier()
      clf.fit(X, y)
      myscore = clf.score(X_test, y_test)
      print("Classifier Score")
      print(myscore)
      

      for k in range(0, len(symptoms)):
        for z in psymptoms:
            if(z == symptoms[k]):
                l2[k] = 1

      inputtest = [l2]
      predict = clf.predict(inputtest)
      predicted = predict[0]


      h = 'no'
      for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

      if (h == 'yes'):
        print(psymptoms)
        print(disease[a])
      else:
        print(psymptoms)
        print("Disease Not Found")          





    return render(request, 'bonge/home.html', context)


def predictdisease(request, model_dir=None):
      # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")

    return JsonResponse("Model Loaded")


def about(request):
  return render(request, 'bonge/about.html')



def team(request):
  return render(request, 'bonge/team.html')




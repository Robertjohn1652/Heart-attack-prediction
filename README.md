#import libraries and modules
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from tkinter import *
import tkinter as tk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time

root = tk.Tk()
root.geometry("1366x800+0+0")
root.title("Heart Attack Detection")
root.iconbitmap("hospital.ico")

LeftFrame = Frame(root, width=650, height=350)
LeftFrame.pack(side='left', fill='both', padx=10, pady=20, expand=True )

# Adding background image

bg = PhotoImage(file = 'hi2.png')

labs = Label(LeftFrame, image=bg)
labs.place(x=0, y=0)

time_label = tk.Label(LeftFrame, bg='#4F7942', width=13, foreground="white",  font=('Helvetica', 16))
time_label.place(x=600, y=5)
def update_time():
    current_time = time.strftime('%H:%M:%S')
    current_date = time.strftime('%d-%m-%Y')
    time_label.config(text=f"{current_time}\n{current_date}")
    LeftFrame.after(1000, update_time) # call this function again in 1000ms

update_time()


df = pd.read_csv("heart.csv")

#datatset source https://www.kaggle.com/datasets/sundayt/heartcsv

# Splitting the dataset into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the models
rfc = RandomForestClassifier()
xgb_model = xgb.XGBClassifier()
logreg = LogisticRegression(max_iter=1000)
nb = GaussianNB()


# Define a preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # normalization
    ('pca', PCA(n_components=13))  # dimensionality reduction
])

# Fit the pipeline on the training data
preprocessing_pipeline.fit(X_train)

# Apply the pipeline on the test data
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Ensure the same preprocessing pipeline is used
# In the model training and evaluation process
X_train_preprocessed = preprocessing_pipeline.transform(X_train)
np.random.seed(42)
# Define the number of folds for cross-validation
n_splits = 5

# Initialize the cross-validator
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define a list to store the F1 scores for each fold
f1_scores = []

# Iterate over the folds
for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    # Split the data into training and test sets for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Fit the model on the training data for this fold
    rfc.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    nb.fit(X_train, y_train)

    # Assume X_test is a pandas dataframe with 13 features
    X_test = X_test.iloc[:, :13]
    # Now the input data has only 10 features and can be used for prediction
    y_pred = rfc.predict(X_test)
    rfc_pred = rfc.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    logreg_pred = logreg.predict(X_test)
    nb_pred = nb.predict(X_test)

    # Calculate F1 scores
    rfc_f1 = f1_score(y_test, rfc_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    logreg_f1 = f1_score(y_test, logreg_pred)
    nb_f1 = f1_score(y_test, nb_pred)
    # Store the F1 score for this fold
    f1_scores.append(rfc_f1)
    f1_scores.append(xgb_f1)
    f1_scores.append(logreg_f1)
    f1_scores.append(nb_f1)




#======================================================
# Calculate accuracy scores
rfc_acc = accuracy_score(y_test, rfc_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
logreg_acc = accuracy_score(y_test, logreg_pred)
nb_acc = accuracy_score(y_test, nb_pred)

# Calculate confusion matrices
rfc_cm = confusion_matrix(y_test, rfc_pred)
xgb_cm = confusion_matrix(y_test, xgb_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)
nb_cm = confusion_matrix(y_test, nb_pred)

# Calculate ROC curve and AUC score
rfc_prob = rfc.predict_proba(X_test)
rfc_fpr, rfc_tpr, rfc_thresholds = roc_curve(y_test, rfc_prob[:, 1])
rfc_auc = auc(rfc_fpr, rfc_tpr)

xgb_prob = xgb_model.predict_proba(X_test)
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_prob[:, 1])
xgb_auc = auc(xgb_fpr, xgb_tpr)

logreg_prob = logreg.predict_proba(X_test)
logreg_fpr, logreg_tpr, logreg_thresholds = roc_curve(y_test, logreg_prob[:, 1])
logreg_auc = auc(logreg_fpr, logreg_tpr)

nb_prob = nb.predict_proba(X_test)
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, nb_prob[:, 1])
nb_auc = auc(nb_fpr, nb_tpr)


#Creating a figure for the ROC curve
fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("ROC Curve")

#Plotting the ROC curves for the models
ax.plot(rfc_fpr, rfc_tpr, label=f"Random Forest (AUC = {rfc_auc:.2f})", linewidth=2)
ax.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.2f})", linewidth=2)
ax.plot(logreg_fpr, logreg_tpr, label=f"Logistic Regression (AUC = {logreg_auc:.2f})", linewidth=2)
ax.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_auc:.2f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='r')
ax.legend()
# Creating a tkinter canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=LeftFrame)
canvas.draw()
canvas.get_tk_widget().place(x=15, y=20)



title_label = Label(LeftFrame, text="Model Evaluation Metrics", font=("Arial", 20), bg='#4F7942', fg='white')
title_label.place(x=1000, y=10)

#Creating labels for displaying the F1-scores
rfc_f1_label = Label(LeftFrame, text=f"Random Forest F1-Scores: {rfc_f1:.2f}", font=("Arial", 14), bg='skyblue', fg='white')
rfc_f1_label.place(x=300, y=550)

xgb_f1_label = Label(LeftFrame, text=f"XGBoost F1-Scores: {xgb_f1:.2f}", font=("Arial", 14), bg='gold', fg='white')
xgb_f1_label.place(x=50, y=550)

logreg_f1_label = Label(LeftFrame, text=f"Logistic Regression AF1-Scores: {logreg_f1:.2f}", font=("Arial", 14), bg='#4F7942', fg='white')
logreg_f1_label.place(x=300, y=600)

nb_f1_label = Label(LeftFrame, text=f"Naive Bayes F1-Scores: {nb_f1:.2f}", font=("Arial", 14), bg='red', fg='white')
nb_f1_label.place(x=50, y=600)
#=============================================================================================================================================

#Creating labels for displaying the accuracy scores
rfc_label = Label(LeftFrame, text=f"Random Forest Accuracy: {rfc_acc:.2f}", font=("Arial", 14), bg='skyblue', fg='white')
rfc_label.place(x=1000, y=60)

xgb_label = Label(LeftFrame, text=f"XGBoost Accuracy: {xgb_acc:.2f}", font=("Arial", 14), bg='gold', fg='white')
xgb_label.place(x=1000, y=110)

logreg_label = Label(LeftFrame, text=f"Logistic Regression Accuracy: {logreg_acc:.2f}", font=("Arial", 14), bg='#4F7942', fg='white')
logreg_label.place(x=1000, y=160)

nb_label = Label(LeftFrame, text=f"Naive Bayes Accuracy: {nb_acc:.2f}", font=("Arial", 14), bg='red', fg='white')
nb_label.place(x=1000, y=210)

#Creating labels for displaying the confusion matrices
rfc_cm_label = Label(LeftFrame, text=f"Random Forest Confusion Matrix:\n{rfc_cm}", font=("Arial", 14), bg='skyblue', fg='white')
rfc_cm_label.place(x=1000, y=260)

xgb_cm_label = Label(LeftFrame, text=f"XGBoost Confusion Matrix:\n{xgb_cm}", font=("Arial", 14), bg='gold', fg='white')
xgb_cm_label.place(x=1000, y=350)

logreg_cm_label = Label(LeftFrame, text=f"Logistic Regression Confusion Matrix:\n{logreg_cm}", font=("Arial", 14), bg='#4F7942', fg='white')
logreg_cm_label.place(x=1000, y=440)

nb_cm_label = Label(LeftFrame, text=f"Naive Bayes Confusion Matrix:\n{nb_cm}", font=("Arial", 14), bg='red', fg='white')
nb_cm_label.place(x=1000, y=540)

def prediction_page():
    root.destroy()
predictionPageLabel = Label(LeftFrame, text="Click the button above to enter the patient's details to be predicted", font=("Arial", 10), bg='green', fg='white')
predictionPageLabel.place(x=640, y=640)

# Create exit button
prediction_page_button = tk.Button(root, text="Click",font=("Arial", 14), bg='green', command=prediction_page)
prediction_page_button.place(x=780, y=620)

root.mainloop()

#===========================================================================================
#===========================================================================================
#============================================================================================
# ===========================================================================================



#PAGE TO ACCEPT AND PREDICT NEW PATIENT'S HEART ATTACK STATUS
#========================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import tkinter as tk
import tkinter as ttk
from tkinter import END
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import time


#====================================================================================================================================
#====================================================================================================================================


# ======================================================================================
warnings.filterwarnings("ignore")

# Load the heart attack dataset
df = pd.read_csv("heart.csv")

# Splitting the dataset into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the models
rfc =  RandomForestClassifier()


# Train and evaluate each model
rfc.fit(X_train, y_train)
#=====================================================================================

# Predict using the models
y_pred = rfc.predict(X_test)
#   ======================================================================================
X.columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thall']
#   ===========================================================================================================================
# Defining the tkinter GUI
root = tk.Tk()
root.geometry("1366x800+0+0")
root.title("Heart Attack Detection")
root.iconbitmap("hospital.ico")

#   =========================================================================================================================

# Creating of Frame

LeftFrame = tk.Frame(root, width=650, height=350)
LeftFrame.pack(side='left', fill='both', padx=10, pady=20, expand=True )
#   ==========================================================================================================================
# Adding background image

bg = tk.PhotoImage(file = 'C:/Users/Cyberlord/PycharmProjects/pythonProject/stock/hi2.png')

labs = tk.Label(LeftFrame, image=bg)
labs.place(x=0, y=0)


# Define a preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # normalization
    ('pca', PCA(n_components=13))  # dimensionality reduction
])

# Fit the pipeline on the training data
preprocessing_pipeline.fit(X_train)

# Apply the pipeline on the test data
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Ensure the same preprocessing pipeline is used
# In the model training and evaluation process
X_train_preprocessed = preprocessing_pipeline.transform(X_train)

np.random.seed(42)
# Calculate the F1 score
f1 = f1_score(y_test, y_pred)

# Define the number of folds for cross-validation
n_splits = 5

# Initialize the cross-validator
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define a list to store the F1 scores for each fold
f1_scores = []

# Iterate over the folds
for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    # Split the data into training and test sets for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    # Fit the model on the training data for this fold
    rfc.fit(X_train, y_train)
    X_test = X_test.iloc[:, :13]
    # Now the input data has only 10 features and can be used for prediction
    y_pred = rfc.predict(X_test)

time_label = tk.Label(root, bg='#4F7942', width=13, foreground="white",  font=('Helvetica', 28))
time_label.place(x=1100, y=20)
def update_time():
    current_time = time.strftime('%H:%M:%S')
    current_date = time.strftime('%d-%m-%Y')
    time_label.config(text=f"{current_time}\n{current_date}")
    root.after(1000, update_time) # call this function again in 1000ms
update_time()

# Defining the labels
name_label = tk.Label(LeftFrame, background='#4F7942', text='what is your name?:', font=('Poppins medium', 14))
name_label.place(x=0, y=0)
age_label = tk.Label(LeftFrame, background='#4F7942', text='what is your age?:', font=('Poppins medium', 14))
age_label.place(x=0, y=40)
sex_label = tk.Label(LeftFrame, text="ENTER(1 FOR MAILE 0 FOR FEMALE) :", bg='#4F7942', font=('Poppins medium', 14))
sex_label.place(x=0, y=80)
cp_label = tk.Label(LeftFrame, text="Enter Chest Pain Type (1-4):", bg='#4F7942',  font=('Poppins medium', 14))
cp_label.place(x=0, y=120 )
trtbps_label = ttk.Label(LeftFrame, text="Enter Resting Blood Pressure:", bg='#4F7942',  font=('Times', 14, 'bold'))
trtbps_label.place(x=0, y=160)
chol_label = ttk.Label(LeftFrame, text="Enter Serum Cholesterol:", bg='#4F7942', font=('Times', 14, 'bold'))
chol_label.place(x=0, y=200)
fbs_label = ttk.Label(LeftFrame, text="Fasting Blood Sugar(0 or 1):", bg='#4F7942',  font=('Times', 14, 'bold'))
fbs_label.place(x=0, y=240)
restecg_label = ttk.Label(LeftFrame, text="Resting ECG Results (0-2):", bg='#4F7942', font=('Times', 14, 'bold'))
restecg_label.place(x=0, y=280)
thalachh_label = tk.Label(LeftFrame, text="Maximum Heart Rate:", bg='#4F7942',  font=('Times', 14, 'bold'))
thalachh_label.place(x=0, y=320)
exang_label = tk.Label(LeftFrame, text="Enter Exercise Induced Angina (0 or 1):", bg='#4F7942',  font=('Times', 14, 'bold'))
exang_label.place(x=0, y=360)
oldpeak_label = tk.Label(LeftFrame, text="Enter ST Depression Induced by Exercise:", bg='#4F7942', font=('Times', 14, 'bold'))
oldpeak_label.place(x=0, y=400)
slope_label = tk.Label(LeftFrame, text="Enter the Slope of the Peak Exercise ST Segment (0-2):", bg='#4F7942', font=('Times', 14, 'bold'))
slope_label.place(x=0, y=440)
ca_label = ttk.Label(LeftFrame, text="Enter Number of Major Vessels Colored by ", bg='#4F7942',  font=('Times', 14, 'bold'))
ca_label.place(x=0, y=480)
ca_labelc = tk.Label(LeftFrame, text="Fluoroscopy (0-3):", bg='#4F7942',  font=('Times', 14, 'bold'))
ca_labelc.place(x=0, y=500)
thall_label = tk.Label(LeftFrame, text="Enter Thalassemia (0-3):", bg='#4F7942', font=('Times', 14, 'bold'))
thall_label.place(x=0, y=540)
#   ============================================================

# creating placeholder
#   ===========================================================

def on_entry_click(event):
    if name_entry.get() == 'Patients Name':
        name_entry.delete(0, 'end')
        name_entry.insert(0, '')
def on_entry_leave(event):
    if name_entry.get() == '':
        name_entry.insert(0, 'Patients Name')
name_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
name_entry.insert(0, 'Patients Name')
name_entry.bind('<FocusIn>', on_entry_click)
name_entry.bind('<FocusOut>', on_entry_leave)
name_entry.place(x=450, y=0)


def on_entry_click(event):
    if age_entry.get() == 'Enter age here':
        age_entry.delete(0, 'end')  # delete all the text in the entry
        age_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if age_entry.get() == '':
        age_entry.insert(0, 'Enter age here')  # Insert placeholder text
age_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
age_entry.insert(0, 'Enter age here')  # Add placeholder text
age_entry.bind('<FocusIn>', on_entry_click)
age_entry.bind('<FocusOut>', on_entry_leave)
age_entry.place(x=450, y=40)


def on_entry_click(event):
    if sex_entry.get() == 'Enter 1 or 0':
        sex_entry.delete(0, 'end')  # delete all the text in the entry
        sex_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if sex_entry.get() == '':
        sex_entry.insert(0, ' Enter 1 or 0')  # Insert placeholder text
sex_entry = tk.Entry(LeftFrame, width=18, font=('poppins', 10), justify='center')
sex_entry.insert(0, 'Enter 1 or 0')  # Add placeholder text
sex_entry.bind('<FocusIn>', on_entry_click)
sex_entry.bind('<FocusOut>', on_entry_leave)
sex_entry.place(x=450, y=80)


def on_entry_click(event):
    if cp_entry.get() == 'Chest Pain Type(1-4):':
        cp_entry.delete(0, 'end')  # delete all the text in the entry
        cp_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if cp_entry.get() == '':
        cp_entry.insert(0, 'Chest Pain Type(1-4):')  # Insert placeholder text
cp_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
cp_entry.insert(0, 'Chest Pain Type(1-4):')  # Add placeholder text
cp_entry.bind('<FocusIn>', on_entry_click)
cp_entry.bind('<FocusOut>', on_entry_leave)
cp_entry.place(x=450, y=120)
def on_entry_click(event):
    if trtbps_entry.get() == 'Resting BP:':
        trtbps_entry.delete(0, 'end')  # delete all the text in the entry
        trtbps_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if trtbps_entry.get() == '':
        trtbps_entry.insert(0, 'Resting BP:')  # Insert placeholder text
trtbps_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
trtbps_entry.insert(0, 'Resting BP:')  # Add placeholder text
trtbps_entry.bind('<FocusIn>', on_entry_click)
trtbps_entry.bind('<FocusOut>', on_entry_leave)
trtbps_entry.place(x=450, y=160)


def on_entry_click(event):
    if chol_entry.get() == 'Serum Cholesterol':
        chol_entry.delete(0, 'end')  # delete all the text in the entry
        chol_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if chol_entry.get() == '':
        chol_entry.insert(0, 'Serum Cholesterol')  # Insert placeholder text
chol_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
chol_entry.insert(0, 'Serum Cholesterol')  # Add placeholder text
chol_entry.bind('<FocusIn>', on_entry_click)
chol_entry.bind('<FocusOut>', on_entry_leave)
chol_entry.place(x=450, y=200)

def on_entry_click(event):
    if fbs_entry.get() == 'Fasting Blood Sugar':
        fbs_entry.delete(0, 'end')  # delete all the text in the entry
        fbs_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if fbs_entry.get() == '':
        fbs_entry.insert(0, 'Fasting Blood Sugar')  # Insert placeholder text
fbs_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
fbs_entry.insert(0, 'Fasting Blood Sugar')  # Add placeholder text
fbs_entry.bind('<FocusIn>', on_entry_click)
fbs_entry.bind('<FocusOut>', on_entry_leave)
fbs_entry.place(x=450, y=240)



def on_entry_click(event):
    if restecg_entry.get() == 'ECG result':
        restecg_entry.delete(0, 'end')  # delete all the text in the entry
        restecg_entry.insert(0, '')  # Insert blank for user input
def on_entry_leave(event):
    if restecg_entry.get() == '':
        restecg_entry.insert(0, 'ECG result')  # Insert placeholder text
restecg_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
restecg_entry.insert(0, 'ECG result')  # Add placeholder text
restecg_entry.bind('<FocusIn>', on_entry_click)
restecg_entry.bind('<FocusOut>', on_entry_leave)
restecg_entry.place(x=450, y=280)


def on_entry_click(event):
    if thalachh_entry.get() == 'Max Heart rate':
        thalachh_entry.delete(0, 'end')
        thalachh_entry.insert(0, '')
def on_entry_leave(event):
    if thalachh_entry.get() == '':
        thalachh_entry.insert(0, 'Max Heart rate')
thalachh_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
thalachh_entry.insert(0, 'Max Heart rate')  # Add placeholder text
thalachh_entry.bind('<FocusIn>', on_entry_click)
thalachh_entry.bind('<FocusOut>', on_entry_leave)
thalachh_entry.place(x=450, y=320)


def on_entry_click(event):
    if exang_entry.get() == 'Induced Agina':
        exang_entry.delete(0, 'end')
        exang_entry.insert(0, '')
def on_entry_leave(event):
    if exang_entry.get() == '':
        exang_entry.insert(0, 'Induced Agina')
exang_entry = ttk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center' )
exang_entry.insert(0, 'Induced Agina')
exang_entry.bind('<FocusIn>', on_entry_click)
exang_entry.bind('<FocusOut>', on_entry_leave)
exang_entry.place(x=450, y=360)


def on_entry_click(event):
    if oldpeak_entry.get() == 'Depression Induced Exercise':
        oldpeak_entry.delete(0, 'end')
        oldpeak_entry.insert(0, '')

def on_entry_leave(event):
    if oldpeak_entry.get() == '':
        oldpeak_entry.insert(0, 'Depression Induced Exercise')
oldpeak_entry = tk.Entry(LeftFrame,  width=18, font=('Poppins', 10), justify='center')
oldpeak_entry.insert(0, 'Depression Induced Exercise', )
oldpeak_entry.bind('<FocusOut>', on_entry_leave)
oldpeak_entry.bind('<FocusIn>', on_entry_click)
oldpeak_entry.place(x=450, y=400)


def on_entry_click(event):
    if slope_entry.get() == 'ST segement':
        slope_entry.delete(0, 'end')
        slope_entry.insert(0, '')
def on_entry_leave(event):
    if slope_entry.get() == '':
        slope_entry.insert(0, 'ST segement')
slope_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
slope_entry.insert(0, 'ST segement')
slope_entry.bind('<FocusIn>', on_entry_click)
slope_entry.bind('<FocusOut>', on_entry_leave)
slope_entry.place(x=450, y=440)


def on_entry_click(event):
    if ca_entry.get() == 'Flouroscopy Result':
        ca_entry.delete(0, 'end')
        ca_entry.insert(0, '')
def on_entry_leave(event):
    if ca_entry.get() == '':
        ca_entry.insert(0, 'Flouroscopy Result')
ca_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
ca_entry.insert(0, 'Flouroscopy Result')
ca_entry.bind('<FocusIn>', on_entry_click)
ca_entry.bind('<FocusOut>', on_entry_leave)
ca_entry.place(x=450, y=480)


def on_entry_click(event):
    if thall_entry.get() == 'Enter Thalassemia':
        thall_entry.delete(0, 'end')
        thall_entry.insert(0, '')

def on_entry_leave(event):
    if thall_entry.get() == '':
        thall_entry.insert(0, 'Enter Thalassemia')

thall_entry = tk.Entry(LeftFrame, width=18, font=('Poppins', 10), justify='center')
thall_entry.insert(0, 'Enter Thalassemia')
thall_entry.bind('<FocusIn>', on_entry_click)
thall_entry.bind('<FocusOut>', on_entry_leave)
thall_entry.place(x=450, y=540)

#==========================================================================================


# Create a label to display the result
result_label = tk.Label(LeftFrame, width=30, height=10, text="", font=('Times', 17), bg='white', fg='blue')
result_label.place(x=680, y=10)

# Create a label to display the accuracy
accuracy_label = tk.Label(LeftFrame, text="", font=('Times', 18), bg='white', fg='black')
accuracy_label.place(x=800, y=10)


#=====================================================================================



# getting patient's data to predict
def predict():
    try:
        name = (name_entry.get())
        sex = (sex_entry.get())
        age = float(age_entry.get())
        cp = int(cp_entry.get())
        trtbps = int(trtbps_entry.get())
        chol = int(chol_entry.get())
        fbs = int(fbs_entry.get())
        restecg = int(restecg_entry.get())
        thalachh = int(thalachh_entry.get())
        exang = int(exang_entry.get())
        oldpeak = float(oldpeak_entry.get())
        slope = float(slope_entry.get())
        ca = int(ca_entry.get())
        thall = float(thall_entry.get())

#   ===========================================================================
        # Create a new dataframe with the entered values
        new_data = pd.DataFrame(
            {'age': [age], 'sex': [sex], 'cp': [cp], 'trtbps': [trtbps], 'chol': [chol], 'fbs': [fbs],
             'restecg': [restecg], 'thalachh': [thalachh], 'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
             'ca': [ca], 'thall': [thall]})
#   ============================================================================
        # Make a prediction using the trained model
        prediction = rfc.predict(new_data)
        prediction_prob = rfc.predict_proba(new_data)

        # Calculate the model's accuracy on test data
        accuracy = rfc.score(X_test, y_test)

        # Plot a bar chart of the prediction probabilities and add accuracy as text
        positive_prob = prediction_prob[0][1]
        negative_prob = prediction_prob[0][0]
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5, 3))
        ax1.bar(['Positive', 'Negative'], [positive_prob, negative_prob], color=['red', 'green'])
        ax1.set_xlabel('Prediction')
        ax1.set_ylabel('Probability')
        ax1.set_title('Heart Attack Prediction')
        ax2.bar(['Accuracy'], [accuracy])
        ax2.text(0, 0.5, '{:.2f}%'.format(accuracy * 100), fontsize=10)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(0, 1.2)
        ax2.axis('off')
        ax2.set_title('Percentage accuracy' '\n' '  of the heart attack prediction')
        plt.tight_layout()

        # attach the figure to the LeftFrame using FigureCanvasTk

        # Create a FigureCanvasTkAgg object and draw the figure on it
        canvas = FigureCanvasTkAgg(fig, master=LeftFrame)
        canvas.draw()

        # Get the Tkinter widget for the canvas and place it in the LeftFrame
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(x=690, y=300, width=550, height=350)


        #   ================================================================================

        # Display the prediction on the result label
        if prediction[0] == 1:
            result_label.config(text= "Dear" + " " + name + ", "  "You are likely going to have a heart attack" , fg="red", wraplength=250)

        else:
            result_label.config(text= "Dear" + " " + name + ", " + "You are not likely to have a heart attack", fg="green", wraplength=250)

    except ValueError:
        result_label.config(text="Please enter valid inputs for all fields!", fg="red", wraplength=250)



# button to trigger prediction
predict_button = tk.Button(LeftFrame, text="Predict", bg='red', font=('Times', 18, 'bold'), command=predict)
predict_button.place(x=250, y=600)

#   =======================================================================================================

def on_reset_button_click():
    # Reset the values in the entry fields
    name_entry.delete(0, END)
    name_entry.insert(0, "Patient's Name")
    age_entry.delete(0, END)
    age_entry.insert(0, "Age")
    sex_entry.delete(0, END)
    sex_entry.insert(0, "Sex")
    cp_entry.delete(0, END)
    cp_entry.insert(0, "Chest pain type")
    trtbps_entry.delete(0, END)
    trtbps_entry.insert(0, "Resting BP")
    chol_entry.delete(0, END)
    chol_entry.insert(0, "Cholesterol")
    fbs_entry.delete(0, END)
    fbs_entry.insert(0, "Fasting blood sugar")
    restecg_entry.delete(0, END)
    restecg_entry.insert(0, "Resting ECG results")
    thalachh_entry.delete(0, END)
    thalachh_entry.insert(0, "Maximum heart rate")
    exang_entry.delete(0, END)
    exang_entry.insert(0, "Exercise induced angina")
    oldpeak_entry.delete(0, END)
    oldpeak_entry.insert(0, "ST depression ")
    slope_entry.delete(0, END)
    slope_entry.insert(0, "slope of the peak")
    ca_entry.delete(0, END)
    ca_entry.insert(0, " Flourosopy Result")
    thall_entry.delete(0, END)
    thall_entry.insert(0, "thalassemia type")
    # Reset the text of the labels
    result_label.config(text="")
    accuracy_label.config(text="")
    # Reset the text of the reset button
    reset_button.config(text="Reset")

reset_button = tk.Button(root, text="Reset", bg='white', font=('Times', 18, 'bold'), command=on_reset_button_click)
reset_button.place(x=400, y=620)

def exit_program():
    root.destroy()
# Create exit button
exitProgram_button = tk.Button(root, text="Exit Program",font=("Arial", 14), bg='green', command=exit_program)
exitProgram_button.place(x=500, y=620)

root.mainloop()

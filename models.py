import pickle

def LR():
    with open('pipeline_with_LR_Tfid.pkl', 'rb') as model_file:
        loaded_LR = pickle.load(model_file)
    return loaded_LR

def MNB():
    with open('pipeline_with_MNB_Tfid.pkl', 'rb') as model_file:
        loaded_MNB = pickle.load(model_file)
    return loaded_MNB

def DTC():
    with open('pipeline_with_DTC_Tfid.pkl', 'rb') as model_file:
        loaded_DTC = pickle.load(model_file)
    return loaded_DTC
    
    

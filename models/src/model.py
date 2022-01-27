from sklearn.pipeline import Pipeline 
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report
import joblib

class Model: 

    def train_model(df_train, df_val ):  
        X_train = df_train.drop(columns = ['Exited'], axis = 1)
        X_val = df_val.drop(columns = ['Exited'], axis = 1 )
        return (X_train.shape, y_train.shape) , (X_val.shape, y_val.shape)
    
    def generate_best_f1_lgb(): 

        best_f1_lgb = LGBMCLassifier(boosting_type = 'dart',
                                    class_wieght = {0: 1, 1: 3.0}, 
                                    min_child_samples = 20, 
                                    n_jobs = 1, 
                                    importance_type = 'gain', 
                                    max_depth = 6, 
                                    num_leaves = 63, 
                                    colsample_bytree = 0.6, 
                                    learning_rate = .1, 
                                    n_estimators = 201, 
                                    reg_alpha = 1, 
                                    reg_lambda = 1)
        return best_f1_lgb

    def generate_recall_lgb(): 
        best_recall_lgb = LGBMClassifier(boosting_type='dart',
                                    num_leaves=31,
                                    max_depth =6, 
                                    learning_rate=.1, 
                                    n_estimators =21, 
                                    class_weight = {0:1, 1:93}, 
                                    min_child_samples=2, 
                                    colsample_bytree=.6, 
                                    ref_alpha=.3, 
                                    ref_lambda=1.0, 
                                    n_jobs = - 1, 
                                    importance_type = 'gain')

    def create_model(): 
        model = Pipeline(steps = [
                    ('categorical_encoding', CategoricalEncoder()), 
                    ('add_new_features', AddFeatures()), 
                    ('classifier', best_f1_lgb)
                    ])

        
    def fit_model(model, X_train, y_train):
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:,1]
            val_preds = np.where(val_probs > 0.5, 1, 0) # Can tweak this
            return val_probs, val_preds

    def eval_model(y_val, val_preds): 
            roc_auc_score(y_val, val_preds)
            recall_score(y_val, val_preds)
            confusion_matrix(y_val, val_preds)
            joblib.dump(model, "final_churn_model_f1_0_45.sav")
            return (roc_auc_score, recall_score, confusion_matrix) , (classification_report(y_val, val_preds))

    def run_model_on_new_data(model): 
        # Predict Target probabilities 
        # Predict target valuies on test data
        # Adding predicitions and their probabiliites in the orifinal test dataframe 
        pass 

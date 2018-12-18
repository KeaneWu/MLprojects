# MLprojects
## Project 0: Music Box (software) user churn prediction. (4_feature_label_generation_with_spark-Copy1 and 5_train_model_sklearn)
### Project Objective: To predict whether the user will churn
#### Process: Label generation, frequency feature generation, and modeling.<br />
Data info: <br />
           0. event dataset, 12252920 records, 4 fields (uid|event| song_id|date) <br />
           1. play dataset, 10863763 records, 5 fields (uid|device|song_id|date|play_time|song_length) <br />
Description: <br />
           0. define the user who is active in label window (between 2017-04-29 ~ 2017-05-12) as 0, otherwise 1.<br />
           1. generate the frequency features for one event type and one time window<br />
           2. get the device of each user from play dataset<br />
           3. create a new data frame with all uid and generated features<br />
           4. fit with Logistic Regression, Random Forest, Gradient Boost Trees models, and Neural Network.<br />
           5. the Gradient Boost Tress got the highest AUC score which is 0.85.<br />
           
## Project 1: 
### Project Objective: To predict the rating
#### Process: review preprocessing and modeling.<br />
Data info: <br />
           last_2_years_restaurant_reviews, 640718 records, 11 fields <br />  
Description: <br />
           0. define the target which average rating = 5 as 1, otherwise 0. <br />
           1. apply TfidfVectorizer to covert the reviews to tf-idf matrix. <br />
           2. fit with Naive-Bayes, Logistic Regression, and Random Forest Classifier. <br />
           3. get the positive and negative features from the prediction.

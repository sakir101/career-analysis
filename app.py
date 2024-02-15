
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def careerPredict() :
    data = request.get_json()
    interest = data['interest']
 
    svm_classifier = joblib.load('svm_model.pkl')
    vect = joblib.load('tfidf_vectorizer.pkl')
    
    vector_form = vect.transform([interest])
    class_probabilities = svm_classifier.predict_proba(vector_form)
    classes = svm_classifier.classes_

    # Create a dictionary to store class probabilities
    class_prob_dict = {}
    for i, class_label in enumerate(classes):
        class_prob_dict[class_label] = class_probabilities[0][i]

    # Find the class with the highest probability
    predicted_class = max(class_prob_dict, key=class_prob_dict.get)
    print("Class Probabilities:")
    for class_label, probability in class_prob_dict.items():
        print(f"{class_label}: {probability * 100:.2f}%")
    # print(f"Predicted Class: {predicted_class}")
    num = predicted_class
    print(num)
    result = {"prediction":  int(num)}
    response = {"class_probabilities": {str(key): value * 100 for key, value in class_prob_dict.items()}}

    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)
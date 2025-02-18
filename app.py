from flask import Flask, render_template, request, jsonify
import base64
import pickle
import librosa
import numpy as np
import os

app = Flask(__name__)

# Function to convert audio into MFCC feature vector
def convert_into_vector(sound):
    print("Sound = ",sound)
    y, sr = librosa.load(sound, duration=3, offset=0.5)
    print("y, sr = ", y, sr)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

@app.route("/", methods=['POST', 'GET'])
def welcom():
    if request.method == "POST":
        audio_data = request.form.get('audio_data')
        
        if audio_data:
            if ',' in audio_data:
                header, encoded = audio_data.split(',', 1)
            else:
                encoded = audio_data

            audio_bytes = base64.b64decode(encoded)

            audio_path = "uploaded_audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)

            # Load the pre-trained model
            with open('SED-1(67).pkl', 'rb') as f:
                clf2 = pickle.load(f)

            
            try:
                mfcc = convert_into_vector(audio_path)
                print("mfcc = ",mfcc)
                print("MFCC shape:", mfcc.shape)  # Expected output: (40,)
                input_data = mfcc.reshape(1, -1)  # Adjust based on model type

                # Get predictions
                predictions = clf2.predict(input_data)
                predicted_class = np.argmax(predictions, axis=1)[0]

                # Emotion class labels
                class_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
                return jsonify({"Predicted emotion": class_labels.get(predicted_class, "Unknown")})
            except:
                print("Faild")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

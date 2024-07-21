import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Read csv file containing data.
df = pd.read_csv(r'C:\Users\ch.avinash chowdary\Downloads\Emotion-based-music-recommendation-system-main\Emotion-based-music-recommendation-system-main\muse_v3.csv')

# Renaming columns of dataframe.
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

# Taking out useful columns.
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]

# Sort column based on emotional & pleasant column values.
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Dividing dataframe based on emotional & pleasant values in sorted order.
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Task of function 'fun' is to take list of unique emotions & return dataframe of 30 rows.
def fun(emotion_list):
    if not emotion_list:
        return pd.DataFrame()

    data = pd.DataFrame()
    t = 30 // len(emotion_list)  # Number of rows for each emotion

    for emotion in emotion_list:
        if emotion == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)])
        elif emotion == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)])
        elif emotion == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)])
        elif emotion == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)])
        else:
            data = pd.concat([data, df_sad.sample(n=t)])
    return data

# Task of function 'pre' is to take list of emotions (containing duplicates also) &
# return unique list of emotion in sorted order based on count.
def pre(emotion_list):
    result = [item for items, c in Counter(emotion_list).most_common()
              for item in [items] * c]

    unique_list = []
    for x in result:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

# Creating model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Loading weights from pretrained model
model.load_weights(r'C:\Users\ch.avinash chowdary\Downloads\Emotion-based-music-recommendation-system-main\Emotion-based-music-recommendation-system-main\model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

# Text or heading's
st.markdown("<h2 style='text-align: center; color: white;'><b>Emotion based music recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

# Just for indentation
col1, col2, col3 = st.columns(3)

emotion_list = []
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION(Click here)'):
        count = 0
        emotion_list.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face_cascade = cv2.CascadeClassifier(r'C:\Users\ch.avinash chowdary\Downloads\Emotion-based-music-recommendation-system-main\Emotion-based-music-recommendation-system-main\haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            count += 1

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                emotion_list.append(emotion_dict[max_index])
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('x') or count >= 20:
                break

        cap.release()
        cv2.destroyAllWindows()

        emotion_list = pre(emotion_list)

with col3:
    pass

new_df = fun(emotion_list)

st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>", unsafe_allow_html=True)
st.write("---------------------------------------------------------------------------------------------------------------------")

try:
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>""".format(l, i + 1, n), unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>".format(a), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except:
    pass

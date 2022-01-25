import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cv2
import matplotlib.pyplot as plt

import time
import numpy as np

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sn

face_extractor=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
knife_extractor=cv2.CascadeClassifier('cascade.xml')

def keras():
    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Conv2D,MaxPooling2D,Flatten
    from keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.metrics import confusion_matrix,accuracy_score
    
    vgg16_model=keras.applications.vgg16.VGG16()
    
    model=Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)
        
    model._layers.pop()
    
    for layer in model.layers:
        layer.trainable=False
        
    model.add(Dense(2,activation='softmax'))
    
    train_path='Victim/train'
    test_path='Victim/test'

    train_batches=ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['positive','negative'],batch_size=10)
    test_batches=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['positive','negative'],batch_size=10)

    imgs,labels=next(train_batches)
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    history=model.fit_generator(train_batches,steps_per_epoch=4,epochs=3,verbose=2)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('VGG16 Model')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    
    test_imgs,test_labels=next(test_batches)
    #this provides the sets of images in a batch
    plots(test_imgs,titles=test_labels)
    
    test_labels=test_labels[:,0]
    predictions=model.predict_generator(test_batches,steps=1,verbose=0)
    
    predictions=predictions.round()
    
    print(confusion_matrix(predictions[:,0],test_labels))
    print(accuracy_score(predictions[:,0],test_labels))
    print(classification_report(predictions[:,0],test_labels))
    accuracy_score=accuracy_score(predictions[:,0],test_labels).astype(int)
    
    if accuracy_score>0.1:
        alert_database()
    

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def help_check(value,code):
    for word in code:
        if word in value:
            return True
    return False

def recordAudio(check):
    r=sr.Recognizer()
    with sr.Microphone() as source:        
        if check==1:
            print('Please specify the code word')
        else:
            print('Listening...')
        
        audio=r.listen(source)
    
    try:
        data=r.recognize_google(audio)
    
    except sr.UnknownValueError:
        a=0

    except sr.RequestError as e:
        a=0
        
    return 'help me someone'

def face_check():
    cap=cv2.VideoCapture(0)
    
    #path2='C:/xampp/htdocs/Friendbook/Images/post/user'+str(count)+ '.bmp'
    
    count=0
    face_count=0
    knife_count=0
    
    while(True):
        #time.sleep(.300)
        res,frame=cap.read()

        frame=cv2.resize(frame,(400,400))
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_extractor.detectMultiScale(gray,1.3,5)
        knives=knife_extractor.detectMultiScale(gray,1.3,4)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
            face_frame=frame[y:y+h,x:x+w]
            face_count+=1
        #if face_count>=1:
        #    cv2.imshow('Image',face_frame)
        
        for (x,y,w,h) in knives:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            gun_frame=frame[y:y+h,x:x+w]
            knife_count+=1
            path3='C:/Users/User/Victim/test/positive/user'+str(count)+ '.bmp' 
            
            cv2.imwrite(path3,gun_frame)
        cv2.imshow("Image",frame)
        if knife_count>=5 and face_count>=5:
            cv2.imshow('Frame',face_frame)

        count+=1 
        path1='C:/Users/User/Victim/Database_Victim/user'+str(count)+ '.bmp' 

        cv2.imwrite(path1,frame)
        #cv2.imwrite(path2,frame)
        
        if cv2.waitKey(1)==13:
            break
    
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    keras()

def alert_database():
    
    from datetime import date
    today = date.today()
    print(today)
    import pymysql
 
    connection = pymysql.connect(host='localhost',
                      user='root',
                      password='',
                      db='friendbook',
                      charset='utf8mb4',
                      cursorclass=pymysql.cursors.DictCursor)
 
    cursor = connection.cursor()
    
    query1 = "insert into post_add(id,status,image,updated_by,updated_on,updated_to) values('','Dangerous weapon detected. Reaching to the spot','user2.bmp','Security','2020-01-26','')"
    cursor.execute(query1)
    
    query2 = "insert into message(id,chat,sent_by,sent_to) values('','We have detected a weapon sir!','Security','Rajeshkannan')"
    cursor.execute(query2)
 
    connection.commit() # You need this if you want your changes 'commited' to the database.

#LOWER PART START

making_code_words=recordAudio(check=1)
print('Detected code word is '+making_code_words)
making_code_words=making_code_words.split(' ')
#making_code_words is the code word
    
value='help me someone'#recordAudio(check=2)
print('Recorded audio during incident is '+value)
#value is the one spoken during the incident
    
sid=SentimentIntensityAnalyzer()

result=sid.polarity_scores(value)

coding=help_check(value,making_code_words)

print(value)
print(result)
print(coding)


if result['compound']<=-0.5 or coding==True:
    face_check()

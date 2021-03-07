import speech_recognition as sr



def speech_to_text(recognizer,mic):


    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration = 1)
        audio = recognizer.listen(source)



    try:
        text = recognizer.recognize_google(audio)
       
    except:
        print("Sorry could not recognize what you said")
    return text    
    
    





def recognizer_source(index):


    r = sr.Recognizer()
    mic=sr.Microphone(device_index=index)

    

    return r, mic





def find_microphone():

   
 for index, name in enumerate(sr.Microphone.list_microphone_names()):
     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

 return









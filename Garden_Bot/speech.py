from gtts import gTTS 
import os 





def text_speech(text):

    language='en'

    myobj = gTTS(text=text, lang=language, slow=False) 

    myobj.save("file.mp3")

    os.system("mpg321 file.mp3")

    os.remove("file.mp3")






    return text
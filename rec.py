# import speech_recognition as sr

# def take_command():#convert the audio to text
#     try:

#         r = sr.Recognizer()
#         with sr.Microphone() as source:
#             r.adjust_for_ambient_noise(source)
#             print('listerning....')
#             audio = r.listen(source)
#             text = r.recognize_google(audio)
#             print('you said: ' + text)
#             return (text.lower())

#     except:
#         print('unable to recognize voice')
#         return 'sr030'
    
# while 1:
#     take_command()

import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Adjusting noise ")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Recording for 4 seconds")
    recorded_audio = recognizer.listen(source, timeout=4)
    print("Done recording")

try:
    print("Recognizing the text")
    text = recognizer.recognize_google(
            recorded_audio, 
            language="en-US"
        )

    print("Decoded Text : {}".format(text))

except Exception as ex:

    print(ex)

